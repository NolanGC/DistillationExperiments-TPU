# ---------------------------------------------------------------------------- #
#                                general imports                               #
# ---------------------------------------------------------------------------- #
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
import torchvision
import argparse
from torchvision import datasets, transforms
# ---------------------------------------------------------------------------- #
#                                module imports                                #
# ---------------------------------------------------------------------------- #
from models import PreResnet, ClassifierEnsemble
from dataloaders import DistillLoader, PermutedDistillLoader
from data import get_dataset
from lossfns import ClassifierTeacherLoss, ClassifierEnsembleLoss, TeacherStudentFwdCrossEntLoss, ClassifierStudentLoss
from training import eval_epoch, supervised_epoch, distillation_epoch
from fileutil import Platform
# ---------------------------------------------------------------------------- #
#                                   CLI args                                   #
# ---------------------------------------------------------------------------- #
#parser = argparse.ArgumentParser()
#parser.add_argument('--permuted', action='store_true', help='permuted argument')
#parser.add_argument('--loadTeachers', action='store_true', help='load teachers from state dict argument')
#parser.add_argument("-d", "--directory", required=True, help="directory path")
#args = parser.parse_args()
dataset_dir = 'data/datasets'
#output_dir = args.directory
# ---------------------------------------------------------------------------- #
#                               experiment flags                               #
# ---------------------------------------------------------------------------- #


def save_object(object, path):
    Platform.save_model(object, 'gs://tianjin-distgen/nolan/' + path)

def load_object(path):
    Platform.load_model('gs://tianjin-distgen/nolan/' + path)

def save_checkpoint(stage, teachers, student, epoch, optimizer):
    """
    Saves the program checkpoint including the following parameters:
    Stage (string) : represents the current stage of the program, either 'teacher0', 'teacher1', 'teacher2' or 'student'
    Teachers (list) : list of the teacher models state_dictionary
    Student (model) : student model state_dictionary
    Epoch (int) : the current epoch of the program
    Optimizer (optimizer) : the optimizer state_dictionary
    """
    checkpoint_object = {
        'stage': stage,
        'teachers': [teacher.state_dict() for teacher in teachers],
        'student': student.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict()
    }

    save_object(checkpoint_object, 'checkpoint.pt')


FLAGS = {}
FLAGS['batch_size'] = 16
FLAGS['num_workers'] = 8 #TODO from XLA example, verify this number is optimal
FLAGS['learning_rate'] = 5e-2
FLAGS['momentum'] = 0.9
FLAGS['weight_decay'] = 1e-4
FLAGS['nestrov'] = True
FLAGS['teacher_epochs'] = 200
FLAGS['student_epochs'] = 300
FLAGS['ensemble_size'] = 3
FLAGS['cosine_annealing_etamin'] = 1e-6
FLAGS['evaluation_frequency'] = 10 # every 10 epochs
FLAGS['permuted'] = False
def main(rank):
    SERIAL_EXEC = xmp.MpSerialExecutor()

    train_dataset, test_dataset = SERIAL_EXEC.run(get_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False)
    train_loader = torch.utils.data.DataLoader(
          train_dataset,
          batch_size=FLAGS['batch_size'],
          sampler=train_sampler,
          num_workers=FLAGS['num_workers'],
          drop_last=True)
    test_loader = torch.utils.data.DataLoader(
          test_dataset,
          batch_size=FLAGS['batch_size'],
          sampler=test_sampler,
          shuffle=False,
          num_workers=FLAGS['num_workers'],
          drop_last=True)
    learning_rate = FLAGS['learning_rate'] * xm.xrt_world_size()
    device = xm.xla_device()

    # load most recent checkpoint
    current_checkpoint = load_object('checkpoint.pt')
    stage = current_checkpoint['stage']
    current_teacher_index = 0

    teachers = [PreResnet(depth=56).to(device) for i in range(FLAGS['ensemble_size'])]
    optimizers = [optim.SGD(teacher.parameters(), lr=learning_rate, momentum=FLAGS['momentum'], weight_decay=FLAGS['weight_decay'], nesterov=FLAGS['nestrov']) for teacher in teachers]
    if(stage == 'teacher0'):
        current_teacher_index = 0
        teachers[0].load_state_dict(current_checkpoint['teachers'][0])
        optimizers[0].load_state_dict(current_checkpoint['optimizer'])
    elif(stage == 'teacher1'):
        current_teacher_index = 1
        teachers[0].load_state_dict(current_checkpoint['teachers'][0])
        teachers[1].load_state_dict(current_checkpoint['teachers'][1])
        optimizers[1].load_state_dict(current_checkpoint['optimizer'])
    elif(stage == 'teacher2'):
        current_teacher_index = 2
        teachers[0].load_state_dict(current_checkpoint['teachers'][0])
        teachers[1].load_state_dict(current_checkpoint['teachers'][1])
        teachers[2].load_state_dict(current_checkpoint['teachers'][2])
        optimizers[2].load_state_dict(current_checkpoint['optimizer'])
    
    if(not stage == 'student'):
        for teacher_index in range(current_teacher_index, FLAGS['ensemble_size']):
            xm.master_print(f"training teacher {teacher_index}")
            model = teachers[teacher_index]
            optimizer = optimizers[teacher_index]
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=FLAGS['teacher_epochs'], eta_min=FLAGS['cosine_annealing_etamin'])
            teacher_loss_fn = ClassifierTeacherLoss(model, device)
            records = []
            eval_metrics = eval_epoch(model, test_loader, epoch=0, device=device, loss_fn=teacher_loss_fn)
            records.append(eval_metrics)
            xm.master_print(f"initial eval metrics:", eval_metrics)
            xm.master_print('<-- training begin -->')
            start_epoch = 0
            if(stage[-1] == str(teacher_index)):
                start_epoch = current_checkpoint['epoch']
            for epoch in range(start_epoch, FLAGS['teacher_epochs']):
                metrics = {}
                train_metrics = supervised_epoch(model, train_loader, optimizer, lr_scheduler, device=device, epoch=epoch+1, loss_fn = teacher_loss_fn)
                metrics.update(train_metrics)
                if(epoch % FLAGS['evaluation_frequency'] == 0):
                    #saving checkpoint
                    save_checkpoint(
                        stage=f'teacher{teacher_index}',
                        teachers=teachers,
                        student=None,
                        epoch=epoch,
                        optimizer=optimizer
                    )
                    eval_metrics = eval_epoch(model, test_loader, device=device, epoch=epoch+1, loss_fn=teacher_loss_fn)
                    metrics.update(eval_metrics)
                records.append(metrics)
                xm.master_print(f"teacher {teacher_index} epoch {epoch} metrics: {metrics}")
            teachers.append(model)
            xm.rendezvous("finalize")
    teacher = ClassifierEnsemble(*teachers)
    if xm.is_master_ordinal():
        Platform.save_model(teachers[0].cpu().state_dict(), 'gs://tianjin-distgen/nolan/NON-PERMUTE_single_teacher_model.pt')
        Platform.save_model(teacher.cpu().state_dict(), 'gs://tianjin-distgen/nolan/NON_PERMUTE_ensemble_teacher_model.pt')
    """
    ------------------------------------------------------------------------------------
    Distilling Data Preparation + Collect Initial Metrics
    ------------------------------------------------------------------------------------
    """

    distill_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
    if FLAGS['permuted']:
        distill_loader = PermutedDistillLoader(temp=4.0, batch_size=FLAGS['batch_size'], shuffle=True, drop_last=True, device=device, sampler=distill_sampler, num_workers=FLAGS['num_workers'], teacher=teacher, dataset=train_dataset)
    else:
        distill_loader = DistillLoader(temp=4.0, batch_size=FLAGS['batch_size'], shuffle=True, drop_last=True, device = device, sampler=distill_sampler, num_workers=FLAGS['num_workers'], teacher=teacher, dataset=train_dataset)
    print(list(teacher.state_dict().values())[0].device)
    #teacher_train_metrics = eval_epoch(teacher, distill_loader, device=device, epoch=0,
                                               #loss_fn=ClassifierEnsembleLoss(teacher, device), isDistillation=True)
    #teacher_test_metrics = eval_epoch(teacher, test_loader, device=device, epoch=0,
                                              #loss_fn=ClassifierEnsembleLoss(teacher, device))
    """
    ------------------------------------------------------------------------------------
    Distilling Student Model
    ------------------------------------------------------------------------------------
    """
    print('distilliing')
    student = PreResnet(deptch=56).to(device)
    optimizer = torch.optim.SGD(params= student.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'], momentum=FLAGS['momentum'], nesterov=FLAGS['nestrov'])
    if(stage == 'student'):
        student.load_state_dict(current_checkpoint['teacher'])
        optimizer.load_state_dict(current_checkpoint['optimizer'])
    
    start_epoch = current_checkpoint['epoch']
    student_base_loss = TeacherStudentFwdCrossEntLoss()
    student_loss = ClassifierStudentLoss(student, student_base_loss, alpha=0.0, device=device) # alpha is set to zero
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=FLAGS['teacher_epochs'], eta_min=FLAGS['cosine_annealing_etamin'])
    records = []
    eval_metrics = eval_epoch(student, test_loader, device=device, epoch=0, loss_fn=student_loss, teacher=teacher)
    records.append(eval_metrics)
    for epoch in range(start_epoch, FLAGS['student_epochs']):
      metrics = {}
      train_metrics = distillation_epoch(student, distill_loader, optimizer,
                                            lr_scheduler, epoch=epoch + 1, loss_fn=student_loss, device=device, dataset=train_dataset, drop_last=True, sampler=distill_sampler, num_workers=FLAGS['num_workers'])
      metrics.update(train_metrics)
      if(epoch % FLAGS['evaluation_frequency'] == 0):
          eval_metrics = eval_epoch(student, test_loader, device=device, epoch=epoch + 1, loss_fn=student_loss, teacher=teacher)
          metrics.update(eval_metrics)
          # save checkpoint
          save_checkpoint(
                stage='student',
                teachers=teachers,
                student=student,
                epoch=epoch,
                optimizer=optimizer
          )
      xm.master_print("student epoch: ", epoch, " metrics: ", metrics)
      records.append(metrics)    
    xm.master_print('done')
    if xm.is_master_ordinal():
        Platform.save_model(student.cpu().state_dict(), 'gs://tianjin-distgen/nolan/NON_PERMUTE_student_model.pt')
    xm.rendezvous("finalize")

if __name__ == "__main__":
    xmp.spawn(main, args=(), nprocs=8, start_method='fork')

