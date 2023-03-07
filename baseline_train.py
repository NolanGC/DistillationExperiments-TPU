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
from data import get_datasetv2
from lossfns import ClassifierTeacherLoss, ClassifierEnsembleLoss, TeacherStudentFwdCrossEntLoss, ClassifierStudentLoss, ClassifierTeacherLossWithTemp
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
    temperature : float
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
    student_learning_rate : float = None

def dict2str(d):
    s = ""
    for k, v in d.items():
        if type(v) is float:
            s = s + f"\t{k}: {str(round(v, 2))}"
        else:
            s = s + f"\t{k}: {v}"
    return s

def main(rank, args):
    SERIAL_EXEC = xmp.MpSerialExecutor()

    train_dataset, test_dataset, valid_dataset = SERIAL_EXEC.run(get_datasetv2)

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
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
          valid_dataset,
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
    valid_loader = torch.utils.data.DataLoader(
          valid_dataset,
          batch_size=args.batch_size,
          sampler=valid_sampler,
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
        records = {"train":[], "test":[], "valid":[]}

        ckpt_path = os.path.join(gcp_root, args.experiment_name, f"teacher-{teacher_index}.ckpt.pt")
        if Platform.exists(ckpt_path):
            ckpt = Platform.load_model(ckpt_path)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            start_epoch = ckpt["next_epoch"]
            records = ckpt["records"]

        teacher_loss_fn = ClassifierTeacherLoss(model, device)

        for split, loader in zip(["test", "valid"], [test_loader, valid_loader]):
            eval_metrics = eval_epoch(model, loader, epoch=start_epoch, device=device, loss_fn=teacher_loss_fn)
            xm.master_print(f"teacher_{teacher_index} initial {split}\t{dict2str(eval_metrics)}")
            records[split].append(eval_metrics)

        for epoch in range(start_epoch, min(args.early_stop_epoch, args.teacher_epochs)):
            train_metrics = supervised_epoch(model, train_loader, train_sampler, optimizer, lr_scheduler,
                device=device, epoch=epoch, loss_fn = teacher_loss_fn)
            records["train"].append(train_metrics)
            if ((epoch + 1) % args.evaluation_frequency == 0):

                for split, loader in zip(["test", "valid"], [test_loader, valid_loader]):
                    eval_metrics = eval_epoch(model, loader, epoch=epoch, device=device, loss_fn=teacher_loss_fn)
                    xm.master_print(f"teacher_{teacher_index} {split}\t{dict2str(eval_metrics)}")
                    records[split].append(eval_metrics)

                Platform.save_model({
                        "model":model.state_dict(),
                        "lr_scheduler":lr_scheduler.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "next_epoch":epoch+1,
                        "records":records
                    },
                    ckpt_path
                )
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

    xm.master_print("Beginning distillation stage.")
    student = teachers[0]
    student_learning_rate = args.student_learning_rate if args.student_learning_rate else args.learning_rate
    optimizer = torch.optim.SGD(params= student.parameters(), lr=student_learning_rate, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=args.nesterov)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.student_epochs, eta_min=args.cosine_annealing_etamin)
    start_epoch = 0
    records = {"train":[], "test":[], "valid":[]}

    ckpt_path = os.path.join(gcp_root, args.experiment_name, f"student.ckpt.pt")
    if Platform.exists(ckpt_path):
        ckpt = Platform.load_model(ckpt_path)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["next_epoch"]
        records = ckpt["records"]

    student_loss = ClassifierTeacherLossWithTemp(student, device, temp=args.temperature, num_classes=100)

    for split, loader in zip(["test", "valid"], [test_loader, valid_loader]):
        eval_metrics = eval_epoch(student, loader, epoch=start_epoch, device=device, loss_fn=student_loss)
        xm.master_print(f"student initial {split}\t{dict2str(eval_metrics)}")
        records[split].append(eval_metrics)

    for epoch in range(start_epoch, args.student_epochs):
      metrics = {}
      train_metrics = supervised_epoch(model, train_loader, train_sampler, optimizer, lr_scheduler,
                device=device, epoch=epoch, loss_fn = teacher_loss_fn)
      records["train"].append(train_metrics)
      if((epoch + 1) % args.evaluation_frequency == 0):

        for split, loader in zip(["test", "valid"], [test_loader, valid_loader]):
          eval_metrics = eval_epoch(model, loader, epoch=epoch, device=device, loss_fn=teacher_loss_fn)
          xm.master_print(f"student {split}\t{dict2str(eval_metrics)}")
          records[split].append(eval_metrics)

        Platform.save_model({
                "model":model.state_dict(),
                "lr_scheduler":lr_scheduler.state_dict(),
                "optimizer":optimizer.state_dict(),
                "next_epoch":epoch+1,
                "records": records,
            },
            ckpt_path
        )

    xm.master_print("Final student evaluation.")
    final_eval_metrics = eval_epoch(student, test_loader, device=device, epoch=0, loss_fn=student_loss)
    xm.master_print('done')
    Platform.save_model(student.cpu().state_dict(), f'gs://tianjin-distgen/nolan/{args.experiment_name}/final_student.pt')
    Platform.save_model(final_eval_metrics, f'gs://tianjin-distgen/nolan/{args.experiment_name}/final_student_metric.pt')
    Platform.save_model(records, f'gs://tianjin-distgen/nolan/{args.experiment_name}/final_student_records.pt')

    xm.rendezvous("finalize-distillation")

if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    args = parser.parse_args().options
    xmp.spawn(main, args=[args], nprocs=8, start_method='fork')


