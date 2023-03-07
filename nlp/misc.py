import torch
import os
import sys
from typing import Union
import collections
import dataclasses

import torch.nn.functional as F
from contextlib import contextmanager

def kd_loss( student_logits : torch.Tensor, teacher_logits : torch.Tensor, temp : Union[int, float]):
    '''
    Calculate the cross entropy between logits_S and logits_T

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temp: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''

    logits_T = teacher_logits / temp
    logits_S = student_logits / temp
    p_T = F.softmax(logits_T, dim=-1)
    loss = - temp ** 2 * (p_T * F.log_softmax(logits_S, dim=-1)).sum(dim=-1).mean()
    return loss

@contextmanager
def silence(enable:bool = True):
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if enable:
            sys.stdout = devnull
            sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def flatten(obj, parent_key='', sep='_'):
    items = []
    for k, v in obj.items():
        new_key = parent_key + sep + k if parent_key else k
        if dataclasses.is_dataclass(v):
            v = dataclasses.asdict(v)
            items.extend(flatten(v, new_key, sep=sep).items())
        elif isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)