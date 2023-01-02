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
from data import get_dataset
from lossfns import ClassifierTeacherLoss
from training import eval_epoch, supervised_epoch, distillation_epoch
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
FLAGS = {}
FLAGS['batch_size'] = 16
FLAGS['num_workers'] = 8 #TODO from XLA example, verify this number is optimal
FLAGS['learning_rate'] = 5e-2
FLAGS['momentum'] = 0.9
FLAGS['weight_decay'] = 1e-4
FLAGS['nestrov'] = True
FLAGS['teacher_epochs'] = 1
FLAGS['student_epochs'] = 1
FLAGS['ensemble_size'] = 2
FLAGS['cosine_annealing_etamin'] = 1e-6
FLAGS['evaluation_frequency'] = 10 # every 10 epochs
FLAGS['permuted'] = True
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
    teachers = []
    for teacher_index in range(FLAGS['ensemble_size']):
        print(f"training teacher {teacher_index}")
        model = PreResnet(depth=56).to(device)
        optimizer = torch.optim.SGD(params= model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'], momentum=FLAGS['momentum'], nesterov=FLAGS['nestrov'])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=FLAGS['teacher_epochs'], eta_min=FLAGS['cosine_annealing_etamin'])
        teacher_loss_fn = ClassifierTeacherLoss(model, device)
        records = []
        eval_metrics = eval_epoch(model, test_loader, epoch=0, device=device, loss_fn=teacher_loss_fn)
        records.append(eval_metrics)
        print(f"initial eval metrics:", eval_metrics)
        print('<-- training begin -->')
        for epoch in range(FLAGS['teacher_epochs']):
            metrics = {}
            train_metrics = supervised_epoch(model, train_loader, optimizer, lr_scheduler,device=device, epoch=epoch+1, loss_fn = teacher_loss_fn)
            metrics.update(train_metrics)
            if(epoch % FLAGS['evaluation_frequency'] == 0):
                eval_metrics = eval_epoch(model, test_loader, device=device, epoch=epoch+1, loss_fn=teacher_loss_fn)
                metrics.update(eval_metrics)
            records.append(metrics)
            print(f"teacher {teacher_index} epoch {epoch} metrics: {metrics}")
        teachers.append(model)
        xm.rendezvous("finalize")
    teacher = ClassifierEnsemble(*teachers)
    
    """
    ------------------------------------------------------------------------------------
    Distilling Data Preparation + Collect Initial Metrics
    ------------------------------------------------------------------------------------
    """
    distill_splits = [train_dataset] # splits is 0 in default config

    if FLAGS['permuted']:
        distill_loader = PermutedDistillLoader(temp=4.0, batch_size=128, shuffle=True, drop_last=False, teacher=teacher, datasets=distill_splits)
    else:
        distill_loader = DistillLoader(temp=4.0, batch_size=128, shuffle=True, drop_last=False, teacher=teacher, datasets=distill_splits)
        
    
    para_distill_loader = pl.ParallelLoader(distill_loader, [device]).per_device_loader(device)
    teacher_train_metrics = eval_epoch(teacher, para_distill_loader, epoch=0,
                                               loss_fn=ClassifierEnsembleLoss(teacher))
    teacher_test_metrics = eval_epoch(teacher, test_loader, epoch=0,
                                              loss_fn=ClassifierEnsembleLoss(teacher))
    """
    ------------------------------------------------------------------------------------
    Distilling Student Model
    ------------------------------------------------------------------------------------
    """

    student = PreResnet(deptch=56).to(device)
    student_base_loss = TeacherStudentFwdCrossEntLoss()
    student_loss = ClassifierStudentLoss(student, student_base_loss, 0.0) # alpha is set to zero
    optimizer = torch.optim.SGD(params= model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'], momentum=FLAGS['momentum'], nesterov=FLAGS['nestrov'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=FLAGS['teacher_epochs'], eta_min=FLAGS['cosine_annealing_etamin'])
    records = []
    eval_metrics = eval_epoch(student, test_loader, epoch=0, loss_fn=student_loss, teacher=teacher)
    records.append(eval_metrics)
    for epoch in range(FLAGS['student_epochs']):
      metrics = {}
      train_metrics = distillation_epoch(student, distill_loader, optimizer,
                                            lr_scheduler, epoch=epoch + 1, loss_fn=student_loss, freeze_bn=False)
      metrics.update(train_metrics)
      if(epoch % FLAGS['evaluation_frequency'] == 0):
          eval_metrics = eval_epoch(student, test_loader, epoch=epoch + 1, loss_fn=student_loss, teacher=teacher)
          metrics.update(eval_metrics)
      print("student epoch: ", epoch, " metrics: ", metrics)
      records.append(metrics)    
    print('done')

if __name__ == "__main__":
    xmp.spawn(main, args=(), nprocs=8, start_method='fork')



