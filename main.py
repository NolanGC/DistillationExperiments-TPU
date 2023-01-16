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
from dataloaders import DistillLoader, PermutedDistillLoader, UniformDistillLoader
from data import get_dataset
from lossfns import ClassifierTeacherLoss, ClassifierEnsembleLoss, TeacherStudentFwdCrossEntLoss, ClassifierStudentLoss
from training import eval_epoch, supervised_epoch, distillation_epoch
from fileutil import Platform
# ---------------------------------------------------------------------------- #
#                                   CLI args                                   #
# ---------------------------------------------------------------------------- #
dataset_dir = 'data/datasets'
gcp_root = 'gs://tianjin-distgen/nolan/'

# ---------------------------------------------------------------------------- #
#                               experiment flags                               #
# ---------------------------------------------------------------------------- #
@dataclass
class Options:
    temperature : int
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
    uniform : bool = False

    # Apply early stopping to teacher.
    early_stop_epoch : int = 999999999

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
          drop_last=False)
    test_loader = torch.utils.data.DataLoader(
          test_dataset,
          batch_size=args.batch_size,
          sampler=test_sampler,
          shuffle=False,
          num_workers=args.num_workers,
          drop_last=False)
    learning_rate = args.learning_rate
    device = xm.xla_device()

    teachers = [PreResnet(depth=56).to(device) for i in range(args.ensemble_size)]
    for teacher_index in range(0, args.ensemble_size):
        xm.master_print(f"training teacher {teacher_index}")
        model = teachers[teacher_index]
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.teacher_epochs, eta_min=args.cosine_annealing_etamin)
        start_epoch = 0

        ckpt_path = os.path.join(gcp_root, args.experiment_name, f"teacher-{teacher_index}.ckpt.pt")
        if Platform.exists(ckpt_path):
            ckpt = Platform.load_model(ckpt_path)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            start_epoch = ckpt["next_epoch"]

        teacher_loss_fn = ClassifierTeacherLoss(model, device)
        records = []
        eval_metrics = eval_epoch(model, test_loader, epoch=start_epoch, device=device, loss_fn=teacher_loss_fn)
        xm.master_print(f"initial eval metrics:", eval_metrics)
        records.append(eval_metrics)
        for epoch in range(start_epoch, min(args.early_stop_epoch, args.teacher_epochs)):
            metrics = {}
            train_metrics = supervised_epoch(model, train_loader, train_sampler, optimizer, lr_scheduler,
                device=device, epoch=epoch, loss_fn = teacher_loss_fn)
            metrics.update(train_metrics)
            if ((epoch + 1) % args.evaluation_frequency == 0):
                eval_metrics = eval_epoch(model, test_loader, device=device, epoch=epoch, loss_fn=teacher_loss_fn)
                metrics.update(eval_metrics)
                Platform.save_model({
                        "model":model.state_dict(),
                        "lr_scheduler":lr_scheduler.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "next_epoch":epoch+1,
                    },
                    ckpt_path
                )
            records.append(metrics)
            xm.master_print(f"teacher {teacher_index} epoch {epoch} metrics: {metrics}")
    xm.rendezvous("finalize")

    xm.master_print("Single teacher evaluation.")
    single_teacher_metrics = eval_epoch(teachers[0], test_loader, device=device, epoch=0,
                                        loss_fn=teacher_loss_fn)

    teacher = ClassifierEnsemble(*teachers)
    xm.master_print("Teacher evaluation.")
    teacher_metrics = eval_epoch(teacher, test_loader, device=device, epoch=0,
                                 loss_fn=ClassifierEnsembleLoss(teacher, device))

    xm.master_print("Completed teacher evaluation.")
    xm.rendezvous("finalize-teacher-eval")

    xm.master_print("Saving models.")
    Platform.save_model(teachers[0].cpu().state_dict(), f"gs://tianjin-distgen/nolan/{args.experiment_name}/final_single_teacher_model.pt")
    Platform.save_model(teacher.cpu().state_dict(), f"gs://tianjin-distgen/nolan/{args.experiment_name}/final_ensemble_model.pt")
    Platform.save_model(single_teacher_metrics, f"gs://tianjin-distgen/nolan/{args.experiment_name}/final_single_teacher_metric.pt")
    Platform.save_model(teacher_metrics, f"gs://tianjin-distgen/nolan/{args.experiment_name}/final_ensemble_metric.pt")
    xm.master_print("Completed saving models.")

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
        distill_loader = PermutedDistillLoader(temp=args.temperature, batch_size=args.batch_size, shuffle=True, drop_last=True, device=device, sampler=distill_sampler, num_workers=args.num_workers, teacher=teacher, dataset=train_dataset)
    elif(args.uniform):
        distill_loader = UniformDistillLoader(temp=args.temperature, batch_size=args.batch_size, shuffle=True, drop_last=True, device=device, sampler=distill_sampler, num_workers=args.num_workers, teacher=teacher, dataset=train_dataset)
    else:
        distill_loader = DistillLoader(temp=args.temperature, batch_size=args.batch_size, shuffle=True, drop_last=True, device = device, sampler=distill_sampler, num_workers=args.num_workers, teacher=teacher, dataset=train_dataset)
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
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.student_epochs, eta_min=args.cosine_annealing_etamin)
    start_epoch = 0

    ckpt_path = os.path.join(gcp_root, args.experiment_name, f"student.ckpt.pt")
    if Platform.exists(ckpt_path):
        ckpt = Platform.load_model(ckpt_path)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["next_epoch"]

    student_base_loss = TeacherStudentFwdCrossEntLoss()
    student_loss = ClassifierStudentLoss(student, student_base_loss, alpha=0.0, device=device) # alpha is set to zero
    records = []
    eval_metrics = eval_epoch(student, test_loader, device=device, epoch=start_epoch, loss_fn=student_loss, teacher=teacher)
    records.append(eval_metrics)
    for epoch in range(start_epoch, args.student_epochs):
      metrics = {}
      train_metrics = distillation_epoch(student, distill_loader, distill_sampler, optimizer,
                                         lr_scheduler, epoch=epoch, loss_fn=student_loss, device=device, dataset=train_dataset, drop_last=True, sampler=distill_sampler, num_workers=args.num_workers)
      metrics.update(train_metrics)
      if((epoch + 1) % args.evaluation_frequency == 0):
        eval_metrics = eval_epoch(student, test_loader, device=device, epoch=epoch, loss_fn=student_loss, teacher=teacher)
        metrics.update(eval_metrics)
        Platform.save_model({
                "model":model.state_dict(),
                "lr_scheduler":lr_scheduler.state_dict(),
                "optimizer":optimizer.state_dict(),
                "next_epoch":epoch+1,
            },
            ckpt_path
        )
        xm.master_print("student epoch: ", epoch, " metrics: ", metrics)
        records.append(metrics)
    xm.master_print("Final student evaluation.")
    final_eval_metrics = eval_epoch(student, test_loader, device=device, epoch=epoch, loss_fn=student_loss, teacher=teacher)
    xm.master_print('done')
    Platform.save_model(student.cpu().state_dict(), f'gs://tianjin-distgen/nolan/{args.experiment_name}/final_student.pt')
    Platform.save_model(final_eval_metrics, f'gs://tianjin-distgen/nolan/{args.experiment_name}/final_student_metric.pt')

    xm.rendezvous("finalize-distillation")

if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    args = parser.parse_args().options
    xmp.spawn(main, args=[args], nprocs=8, start_method='fork')


