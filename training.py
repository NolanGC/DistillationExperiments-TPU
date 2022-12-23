import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.distributions.kl import kl_divergence
from torch.distributions import Categorical
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from utils import batch_calibration_stats, expected_calibration_err, reduce_ensemble_logits, preact_cka

def get_lr(lr_scheduler):
    return lr_scheduler.get_last_lr()[0]

def supervised_epoch(net, loader, optimizer, lr_scheduler,device, epoch, loss_fn):
    """
    Train the network for one epoch.
    Inputs:
        net: the network to train
        loader: the data loader
        optimizer: the optimizer
        lr_scheduler: the learning rate scheduler
        epoch: the current epoch
        loss_fn: the loss function
    Returns:
        metrics: a dictionary of metrics
    """
    net.train()
    train_loss = torch.tensor(0.).to(device)
    correct = torch.tensor(0.).to(device)
    total = 0
    para_train_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)
    for batch_idx, (inputs, targets) in enumerate(para_train_loader):
        optimizer.zero_grad()
        
        loss, outputs = loss_fn(inputs, targets)
        loss.backward()
        xm.optimizer_step(optimizer)
        outputs = outputs.to(device)
        train_loss += loss
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum()

    lr_scheduler.step()
    metrics = dict(
            train_loss=train_loss / len(loader),
            train_acc=100 * correct / total,
            lr=lr_scheduler.get_last_lr()[0],
            epoch=epoch
        )
    return metrics


def eval_epoch(net, loader, epoch, loss_fn, device=None, teacher=None, with_cka=True):
    """
    Evaluate the model on the test set.
    Inputs: 
        net: the model to evaluate
        loader: the data loader for the test set
        epoch: the current epoch
        loss_fn: the loss function to use
        teacher: the teacher model to use for distillation
        with_cka: whether to compute CKA
    Returns:
        metrics: a dictionary of metrics with format
    """
    net.eval()
    test_loss = torch.tensor(0.).to(device) 
    correct = torch.tensor(0.).to(device)
    total = 0
    agree = torch.tensor(0.).to(device)
    nll = torch.tensor(0.).to(device)
    kl = torch.tensor(0.).to(device)
    ece_stats = None
    para_train_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device) 
    for bath_idx, batch in enumerate(para_train_loader):
        with torch.no_grad():
            # [:2] to ignore teacher logits in the case of distillation
            inputs, targets = batch[:2]
            loss_args = [inputs, targets]
            if teacher is not None:
                teacher_logits = teacher(inputs)
                teacher_logits = reduce_ensemble_logits(teacher_logits)
                loss_args.append(teacher_logits)
            loss, logits = loss_fn(*loss_args)

            test_loss += loss
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum()
            nll += -F.log_softmax(logits, dim=-1)[..., targets].mean()
            if teacher is not None:
                teacher_predicted = teacher_logits.argmax(-1)
                agree += predicted.eq(teacher_predicted).sum()
                kl += kl_divergence(
                    Categorical(logits=teacher_logits),
                    Categorical(logits=logits)
                ).mean()
            batch_ece_stats = batch_calibration_stats(logits, targets, num_bins=10)
            ece_stats = batch_ece_stats if ece_stats is None else [
                t1 + t2 for t1, t2 in zip(ece_stats, batch_ece_stats)
            ]
            xm.mark_step()
    ece = expected_calibration_err(*ece_stats, num_samples=total)
    metrics = dict(
        test_loss=test_loss / len(loader),
        test_acc=100. * correct / total,
        test_ece=ece,
        test_nll=nll / len(loader),
        epoch=epoch,
    )

    # only return generalization metrics if there is no teacher
    if teacher is None:
        print(metrics)
        return metrics

    # add fidelity metrics
    metrics.update(dict(test_ts_agree=100. * agree / total, test_ts_kl=kl / len(loader)))
    if len(teacher.components) == 1 and hasattr(teacher.components[0], 'preacts') and with_cka:
        cka = preact_cka(teacher.components[0], net, loader)
        metrics.update({f'test_cka_{i}': val for i, val in enumerate(cka)})
    print(metrics)
    return metrics


def distillation_epoch(student, train_loader, optimizer, lr_scheduler, epoch,
                       loss_fn):
    student.train()
    train_loss, correct, agree, total, real_total = 0, 0, 0, 0, 0
    kl = 0
    ece_stats = None
    num_batches = len(train_loader)

    for batch_idx, (inputs, targets, teacher_logits, temp) in enumerate(train_loader):
        optimizer.zero_grad()
        loss, student_logits = loss_fn(inputs, targets, teacher_logits, temp)
        loss.backward()
        xm.optimizer_step(optimizer)

        train_loss += loss.item()
        student_predicted = student_logits.argmax(-1)
        teacher_predicted = teacher_logits.argmax(-1)
        full_batch_size = inputs.size(0)
        real_batch_size = targets.size(0)
        total += full_batch_size
        real_total += real_batch_size
        correct += student_predicted[:real_batch_size].eq(targets).sum().item()
        agree += student_predicted.eq(teacher_predicted).sum().item()

        kl += kl_divergence(
            Categorical(logits=teacher_logits),
            Categorical(logits=student_logits)
        ).mean().item()

        batch_ece_stats = batch_calibration_stats(student_logits[:real_batch_size], targets, num_bins=10)
        ece_stats = batch_ece_stats if ece_stats is None else [
            t1 + t2 for t1, t2 in zip(ece_stats, batch_ece_stats)
        ]

    lr_scheduler.step()
    if hasattr(loss_fn.base_loss, 'step'):
        loss_fn.base_loss.step()
    ece = expected_calibration_err(*ece_stats, num_samples=total)
    metrics = dict(
            train_loss=train_loss / num_batches,
            train_acc=100 * correct / real_total,
            train_ts_agree=100 * agree / total,
            train_ece=ece,
            train_ts_kl=kl / num_batches,
            lr=lr_scheduler.get_last_lr()[0],
            epoch=epoch
        )
    # metrics.update(ece_bin_metrics(*ece_stats, num_bins=10, prefix='train'))
    return metrics
