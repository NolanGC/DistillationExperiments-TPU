from torch.utils.data import DataLoader
import torch
import numpy as np
from omegaconf import ListConfig
import perm_utils
from utils import reduce_ensemble_logits

class DistillLoader(object):
    def __init__(self, teacher, datasets, temp, batch_size, shuffle, drop_last, **kwargs):
        if isinstance(temp, ListConfig):
            assert len(temp) == len(datasets)
        if isinstance(temp, float):
            temp = [temp] * len(datasets)

        self.teacher = teacher
        self.temp = temp
        self.batch_size = batch_size

        self.loaders = self._make_loaders(datasets, batch_size, shuffle, drop_last)

    def __len__(self):
        return min([len(ldr) for ldr in self.loaders])

    def __iter__(self):
        return self.generator

    def _make_loaders(self, datasets, total_batch_size, shuffle, drop_last):
        assert self.synth_ratio < 1
        num_real = sum([len(dset) for dset in datasets])
        num_total = int(num_real / (1 - self.synth_ratio))
        b_sizes = [
            int(len(dset) / num_total * total_batch_size) for dset in datasets[:-1]
        ]
        synth_bs = int(self.synth_ratio * total_batch_size)
        b_sizes.append(total_batch_size - sum(b_sizes) - synth_bs)
        loaders = [
            DataLoader(dset, bsize, shuffle, drop_last=drop_last) for dset, bsize in zip(datasets, b_sizes)
        ]
        return loaders

    @property
    def generator(self):
        for batches in zip(*self.loaders):
            bs_list = [b[0].size(0) for b in batches]
            inputs = torch.cat([b[0] for b in batches])
            targets = torch.cat([b[1] for b in batches])

            with torch.no_grad():
                logits = reduce_ensemble_logits(self.teacher(inputs))

            assert len(bs_list) == len(self.temp)
            temp = torch.cat([
                torch.ones(bs, 1) * t for bs, t in zip(bs_list, self.temp)
            ])
            yield inputs, targets, logits, temp
            
class PermutedDistillLoader(DistillLoader):
    def __init__(self, teacher, datasets, temp, batch_size, shuffle, drop_last, **kwargs):
       super(PermutedDistillLoader, self).__init__(teacher, datasets, temp, batch_size, shuffle, drop_last, **kwargs)

    @property
    def generator(self):
        for batches in zip(*self.loaders):
            bs_list = [b[0].size(0) for b in batches]
            inputs = torch.cat([b[0] for b in batches])
            targets = torch.cat([b[1] for b in batches])
            
            with torch.no_grad():
                teacher_logits = reduce_ensemble_logits(self.teacher(inputs))
                batch_size = teacher_logits.shape[0]
                # i.e for cifar100, logit_size = 100
                logit_size = teacher_logits.shape[1]
                # this will store our permuted teacher logits
                # iterate through each batch, permute the logits
                permutation_matrices = perm_utils.batch_permutation_matrix(batch_size, logit_size, targets)
                permuted_teacher_logits = torch.stack([permutation_matrices[i].float() @ teacher_logits[i] for i in range(batch_size)])
            assert len(bs_list) == len(self.temp)
            temp = torch.cat([
                torch.ones(bs, 1) * t for bs, t in zip(bs_list, self.temp)
            ])
            yield inputs, targets, permuted_teacher_logits, temp
