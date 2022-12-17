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
from models import PreResnet
from data import get_dataset
# ---------------------------------------------------------------------------- #
#                                   CLI args                                   #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--permuted', action='store_true', help='permuted argument')
parser.add_argument('--loadTeachers', action='store_true', help='load teachers from state dict argument')
parser.add_argument("-d", "--directory", required=True, help="directory path")
args = parser.parse_args()
dataset_dir = 'data/datasets'
output_dir = args.directory
# ---------------------------------------------------------------------------- #
#                               experiment flags                               #
# ---------------------------------------------------------------------------- #
FLAGS = {}
FLAGS['batch_size'] = 128
FLAGS['num_workers'] = 4 #TODO from XLA example, verify this number is optimal
FLAGS['learning_rate'] = 5e-2
FLAGS['momentum'] = 0.9
FLAGS['weight_decay'] = 1e-4
FLAGS['nestrov'] = True
FLAGS['teacher_epochs'] = 200
FLAGS['ensemble_size'] = 3

SERIAL_EXEC = xmp.MpSerialExecutor()
WRAPPED_MODEL = xmp.MpModelWrapper(PreResnet(depth=56))

train_dataset, test_dataset = SERIAL_EXEC.run(get_dataset)
train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset,
      num_replicas=xm.xrt_world_size(),
      rank=xm.get_ordinal(),
      shuffle=True)
train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=FLAGS['batch_size'],
      sampler=train_sampler,
      num_workers=FLAGS['num_workers'],
      drop_last=True)
test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=FLAGS['batch_size'],
      shuffle=False,
      num_workers=FLAGS['num_workers'],
      drop_last=True)

print('train_loader length: ', len(train_loader))
print('test_loader length: ', len(test_loader))
#learning_rate = FLAGS['learning_rate'] * xm.xrt_world_size()
#device = xm.xla_device()
#model = WRAPPED_MODEL.to(device)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate,
#                      momentum=FLAGS['momentum'], weight_decay=5e-4)

