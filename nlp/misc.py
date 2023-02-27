import torch
import os
import sys

import torch.nn.functional as F
from contextlib import contextmanager

def masked_kd_loss( student_logits, teacher_logits, temperature=1, mask=None):
    '''
    Calculate the cross entropy between logits_S and logits_T

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''

    if mask:
        assert len(mask) == 1
        mask = mask[0].unsqueeze(-1)
    else:
        mask = torch.tensor(1.0)

    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = teacher_logits / temperature
    beta_logits_S = student_logits / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1) * mask).sum(dim=-1)
    loss = loss.sum() / torch.sum(mask)
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