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
from lossfns import ClassifierTeacherLoss, ClassifierEnsembleLoss, TeacherStudentFwdCrossEntLoss, ClassifierStudentLoss, TeacherStudentUniformFwdCrossEntLoss, TeacherStudentUniformArgmaxFwdCrossEntLoss
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
    cosine_annealing_etamin : float
    evaluation_frequency : int
    experiment_name : str
    num_trials: int
    trial_epochs: int
    # Apply early stopping to teacher.
    early_stop_epoch : int = 999999999

def compute_el2n_score(model, num_epochs, example):
    # Compute the output of the model for the example
    output = model(example)
    
    # Compute the squared L2 norm of the output
    l2_norm = torch.norm(output, p=2, dim=1)**2
    
    # Compute the average L2 norm across all examples in the batch
    avg_l2_norm = torch.mean(l2_norm)
    
    # Compute the EL2N score as the average L2 norm over the last `num_epochs` epochs
    el2n_score = avg_l2_norm / num_epochs
    
    return el2n_score.item()

def average_el2n_score(models, num_epochs, example):
    # Compute the EL2N score for each model in the ensemble
    el2n_scores = [compute_el2n_score(model, num_epochs, example) for model in models]
    
    # Compute the average EL2N score across the ensemble
    avg_el2n_score = np.mean(el2n_scores)
    
    return avg_el2n_score

def main(rank, args):
    SERIAL_EXEC = xmp.MpSerialExecutor()

    # initialize datasets
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

    models = [PreResnet(depth=56).to(device) for _ in range(args.num_trials)]
    for trial in range(args.num_trials):
        xm.master_print(f"Starting trial {trial}")
        model = models[trial]
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.teacher_epochs, eta_min=args.cosine_annealing_etamin)
        start_epoch = 0
        ckpt_path = os.path.join(gcp_root, args.experiment_name, f"trial-{trial}.ckpt.pt")
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
        for epoch in range(start_epoch, min(args.early_stop_epoch, args.trial_epochs)):
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
            xm.master_print(f"trial {trial} epoch {epoch} metrics: {metrics}")
    # now calulate EL2N scores using the model after trial_epochs
    for trial, model in enumerate(models):
        Platform.save_model(model.cpu().state_dict(), f'gs://tianjin-distgen/nolan/{args.experiment_name}/model_{trial}.pt')
        xm.master_print(f"Saved model {trial} to gs://tianjin-distgen/nolan/{args.experiment_name}/model_{trial}.pt")
    xm.master_print("Finished training, now calculating EL2N scores")

    sorted_dataset = sorted(train_dataset, key=lambda example: compute_el2n_score(example))
    sorted_dataloader = torch.utils.data.DataLoader(sorted_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    # save dataloader object
    Platform.save_model(sorted_dataloader, f'gs://tianjin-distgen/nolan/{args.experiment_name}/sorted_dataloader.pt')

if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    args = parser.parse_args().options
    xmp.spawn(main, args=[args], nprocs=8, start_method='fork')


