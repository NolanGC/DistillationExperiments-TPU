from torch.utils.data import DataLoader
import torch
import numpy as np
import perm_utils
from utils import reduce_ensemble_logits
import torch_xla.distributed.parallel_loader as pl


class DistillLoader(object):
    def __init__(self, teacher, dataset, temp, batch_size, shuffle, drop_last, device, sampler, num_workers, **kwargs):
        self.teacher = teacher
        self.device = device
        self.temp = temp
        self.batch_size = batch_size
        self.loader = self._make_loader(dataset, drop_last, sampler, num_workers)
        
    def __len__(self):
        return len(self.loader) 

    def __iter__(self):
        return self.generator

    def _make_loader(self, dataset, drop_last, sampler, num_workers):
        loader = pl.ParallelLoader(DataLoader(dataset, self.batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last), [self.device]).per_device_loader(self.device)
        return loader

    @property
    def generator(self):
        self.teacher.to(self.device)
        for inputs, targets in self.loader:
            with torch.no_grad():
                logits = reduce_ensemble_logits(self.teacher(inputs))
            current_batch_size = inputs.shape[0] # protect against uneven division
            temp = torch.cat([
                torch.ones(current_batch_size, 1) * self.temp
            ])
            yield inputs, targets, logits, temp
            
class PermutedDistillLoader(DistillLoader):
    def __init__(self, teacher, dataset, temp, batch_size, shuffle, drop_last, device, **kwargs):
       super(PermutedDistillLoader, self).__init__(teacher, dataset, temp, batch_size, shuffle, drop_last, device,**kwargs)

    @property
    def generator(self):
        self.teacher.to(self.device)
        for inputs, targets in self.loader:
            inputs.to(self.device)
            targets.to(self.device)
            with torch.no_grad():
                teacher_logits = reduce_ensemble_logits(self.teacher(inputs))
                batch_size = inputs.shape[0]
                # i.e for cifar100, logit_size = 100
                logit_size = teacher_logits.shape[1]
                # this will store our permuted teacher logits
                # iterate through each batch, permute the logits
                permutation_matrices = perm_utils.batch_permutation_matrix(batch_size, logit_size, targets).to(self.device)
                permuted_teacher_logits = torch.stack([permutation_matrices[i].float() @ teacher_logits[i] for i in range(batch_size)])
            temp = torch.cat([
                torch.ones(batch_size, 1) * self.temp
            ])
            yield inputs, targets, permuted_teacher_logits, temp

class UniformDistillLoader(DistillLoader):
    def __init__(self, teacher, dataset, temp, batch_size, shuffle, drop_last, device, **kwargs):
        super(UniformDistillLoader, self).__init__(teacher, dataset, temp, batch_size, shuffle, drop_last, device,**kwargs)
    def generator(self):
        self.teacher.to(self.device)
        for inputs, targets in self.loader:
            inputs.to(self.device)
            targets.to(self.device)
            with torch.no_grad():
                teacher_logits = reduce_ensemble_logits(self.teacher(inputs))
                batch_size = inputs.shape[0]
                num_classes = teacher_logits.shape[1]

                # Create one-hot mask
                mask = torch.one_hot(targets, num_classes).to(self.device)

                # Multiply mask with teacher_logits
                logit_of_correct_class = (mask * teacher_logits).sum(dim=1)

                # Create uniform logits
                uniform_logits = (1 - logit_of_correct_class[:, None]) / (num_classes - 1)
                uniform_logits = uniform_logits * (1 - mask) + logit_of_correct_class[:, None] * mask
                uniform_logits.to(self.device)
            temp = torch.cat([
                torch.ones(batch_size, 1) * self.temp
            ])
            yield inputs, targets, uniform_logits, temp

