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
import torch_xla.utils.serialization as xser
import torchvision
import argparse
from torchvision import datasets, transforms
import argparse
from dataclasses import dataclass
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

@dataclass
class Options:
    batch_size : int
    num_workers : int
    learning_rate : float
    momentum : float
    weight_decay : float
    nesterov : bool
    teacher_epochs : int
    student_epochs : int
    ensemble_size : int
    cosine_annealing_etamin : float
    evaluation_frequency : int
    permuted : bool
    experiment_name : str

def load_object(path):
    Platform.copyfile(f"gs://tianjin-distgen/nolan/{args.experiment_name}/" + path, 'temp_checkpointdl.pt')
    obj = xser.load('temp_checkpointdl.pt')
    return obj

def save_checkpoint(stage, teachers, student, epoch, scheduler, optimizer):
    """
    Saves the program checkpoint including the following parameters:
    Stage (string) : represents the current stage of the program, either 'teacher0', 'teacher1', 'teacher2' or 'student'
    Teachers (list) : list of the teacher models state_dictionary
    Student (model) : student model state_dictionary
    Epoch (int) : the current epoch of the program
    Optimizer (optimizer) : the optimizer state_dictionary
    """
    #xm.mark_step()
    #print("OPTIMIZER STATE DICT", optimizer.state_dict())
    #print("OPTIMIZER STATE DICT len", len(list(optimizer.state_dict().keys())))

    checkpoint_object = {
        'stage': stage,
        'teachers': [teacher.state_dict() for teacher in teachers],
        'student': student.state_dict() if student else None,
        'epoch': epoch,
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    xm.save(checkpoint_object, 'checkpoint.pt')
    if(xm.is_master_ordinal()):
        Platform.copyfile('checkpoint.pt', f"gs://tianjin-distgen/nolan/{args.experiment_name}/checkpoint.pt")

def main(rank, args):
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
          batch_size=args.batch_size,
          sampler=train_sampler,
          num_workers=args.num_workers,
          drop_last=True)
    test_loader = torch.utils.data.DataLoader(
          test_dataset,
          batch_size=args.batch_size,
          sampler=test_sampler,
          shuffle=False,
          num_workers=args.num_workers,
          drop_last=True)
    learning_rate = args.learning_rate
    device = xm.xla_device()
    current_checkpoint = None

    stage = 'teacher0'
    teachers = [PreResnet(depth=56).to(device) for i in range(args.ensemble_size)]
    current_teacher_index = 0
    if(Platform.exists(f"gs://tianjin-distgen/nolan/{args.experiment_name}/checkpoint.pt")):
        current_checkpoint = load_object("checkpoint.pt")
        xm.master_print("LOADED CHECKPOINT", current_checkpoint['stage'], current_checkpoint['epoch'])
    else:
        current_checkpoint = None

    if(current_checkpoint):
        stage = current_checkpoint['stage']

        if(stage == 'teacher0'):
            current_teacher_index = 0
            teachers[0].load_state_dict(current_checkpoint['teachers'][0])
        elif(stage == 'teacher1'):
            current_teacher_index = 1
            teachers[0].load_state_dict(current_checkpoint['teachers'][0])
            teachers[1].load_state_dict(current_checkpoint['teachers'][1])
        elif(stage == 'teacher2'):
            current_teacher_index = 2
            teachers[0].load_state_dict(current_checkpoint['teachers'][0])
            teachers[1].load_state_dict(current_checkpoint['teachers'][1])
            teachers[2].load_state_dict(current_checkpoint['teachers'][2])

        optimizer = optim.SGD(teachers[current_teacher_index].parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer.load_state_dict(current_checkpoint['optimizer'])
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.teacher_epochs, eta_min=args.cosine_annealing_etamin)
        lr_scheduler.load_state_dict(current_checkpoint['scheduler'])

    if(not stage == 'student'):
        for teacher_index in range(current_teacher_index, args.ensemble_size):
            xm.master_print(f"training teacher {teacher_index}")
            model = teachers[teacher_index]
            if(not teacher_index == current_teacher_index or not current_checkpoint):
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.teacher_epochs, eta_min=args.cosine_annealing_etamin)
            teacher_loss_fn = ClassifierTeacherLoss(model, device)
            records = []
            xm.master_print('<-- training begin -->')
            start_epoch = 0
            if(current_checkpoint and stage[-1] == str(teacher_index)):
                start_epoch = current_checkpoint['epoch']
            eval_metrics = eval_epoch(model, test_loader, epoch=start_epoch, device=device, loss_fn=teacher_loss_fn)
            xm.master_print(f"initial eval metrics:", eval_metrics)
            records.append(eval_metrics)
            for epoch in range(start_epoch, args.teacher_epochs):
                metrics = {}
                train_metrics = supervised_epoch(model, train_loader, optimizer, lr_scheduler, device=device, epoch=epoch+1, loss_fn = teacher_loss_fn)
                metrics.update(train_metrics)
                if((epoch + 1) % args.evaluation_frequency == 0):
                    eval_metrics = eval_epoch(model, test_loader, device=device, epoch=epoch+1, loss_fn=teacher_loss_fn)
                    metrics.update(eval_metrics)
                #saving checkpoint
                save_checkpoint(
                    stage=f'teacher{teacher_index}',
                    teachers=teachers,
                    student=None,
                    epoch=epoch+1,
                    scheduler=lr_scheduler,
                    optimizer=optimizer
                )
                records.append(metrics)
                xm.master_print(f"teacher {teacher_index} epoch {epoch} metrics: {metrics}")
            xm.rendezvous("finalize")
    teacher = ClassifierEnsemble(*teachers)
    xm.master_print("Teacher evaluation.")
    teacher_metrics = eval_epoch(teacher, test_loader, device=device, epoch=0,
                                      loss_fn=ClassifierEnsembleLoss(teacher, device))

    if xm.is_master_ordinal():
        Platform.save_model(teachers[0].cpu().state_dict(), f"gs://tianjin-distgen/nolan/{args.experiment_name}/final_single_teacher_model.pt")
        Platform.save_model(teacher.cpu().state_dict(), f"gs://tianjin-distgen/nolan/{args.experiment_name}/final_ensemble_model.pt")
        Platform.save_model(teacher_metrics, f"gs://tianjin-distgen/nolan/{args.experiment_name}/final_ensemble_metric.pt")
        teacher_metrics

    teachers = [teacher.to(device) for teacher in teachers]
    teacher.to(device)
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
    if args.permuted:
        distill_loader = PermutedDistillLoader(temp=4.0, batch_size=args.batch_size, shuffle=True, drop_last=True, device=device, sampler=distill_sampler, num_workers=args.num_workers, teacher=teacher, dataset=train_dataset)
    else:
        distill_loader = DistillLoader(temp=4.0, batch_size=args.batch_size, shuffle=True, drop_last=True, device = device, sampler=distill_sampler, num_workers=args.num_workers, teacher=teacher, dataset=train_dataset)

    #teacher_train_metrics = eval_epoch(teacher, distill_loader, device=device, epoch=0,
                                               #loss_fn=ClassifierEnsembleLoss(teacher, device), isDistillation=True)

    """
    ------------------------------------------------------------------------------------
    Distilling Student Model
    ------------------------------------------------------------------------------------
    """
    xm.master_print("Beginning distillation stage.")
    student = PreResnet(deptch=56).to(device)
    optimizer = torch.optim.SGD(params= student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=args.nesterov)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.teacher_epochs, eta_min=args.cosine_annealing_etamin)
    start_epoch = 0
    if(current_checkpoint and current_checkpoint['student']):
        student.load_state_dict(current_checkpoint['student'])
    if(current_checkpoint):
        if(current_checkpoint['stage'] == 'student'):
            start_epoch = current_checkpoint['epoch']
        lr_scheduler.load_state_dict(current_checkpoint['scheduler'])
        optimizer.load_state_dict(current_checkpoint['optimizer'])
    student_base_loss = TeacherStudentFwdCrossEntLoss()
    student_loss = ClassifierStudentLoss(student, student_base_loss, alpha=0.0, device=device) # alpha is set to zero
    records = []
    eval_metrics = eval_epoch(student, test_loader, device=device, epoch=start_epoch, loss_fn=student_loss, teacher=teacher)
    records.append(eval_metrics)
    for epoch in range(start_epoch, args.student_epochs):
      metrics = {}
      train_metrics = distillation_epoch(student, distill_loader, optimizer,
                                            lr_scheduler, epoch=epoch + 1, loss_fn=student_loss, device=device, dataset=train_dataset, drop_last=True, sampler=distill_sampler, num_workers=args.num_workers)
      metrics.update(train_metrics)
      if(epoch % args.evaluation_frequency == 0):
        eval_metrics = eval_epoch(student, test_loader, device=device, epoch=epoch + 1, loss_fn=student_loss, teacher=teacher)
        metrics.update(eval_metrics)
        # save checkpoint
        save_checkpoint(
            stage='student',
            teachers=teachers,
            student=student,
            epoch=epoch,
            scheduler=lr_scheduler,
            optimizer=optimizer
        )
        xm.master_print("student epoch: ", epoch, " metrics: ", metrics)
        records.append(metrics)
    xm.master_print("Final student evaluation.")
    final_eval_metrics = eval_epoch(student, test_loader, device=device, epoch=epoch + 1, loss_fn=student_loss, teacher=teacher)
    xm.master_print('done')
    if xm.is_master_ordinal():
        Platform.save_model(student.cpu().state_dict(), f'gs://tianjin-distgen/nolan/{args.experiment_name}/final_student.pt')
        Platform.save_model(final_eval_metrics, f'gs://tianjin-distgen/nolan/{args.experiment_name}/final_student_metric.pt')
    xm.rendezvous("finalize")

if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    args = parser.parse_args().options
    xmp.spawn(main, args=[args], nprocs=8, start_method='fork')


