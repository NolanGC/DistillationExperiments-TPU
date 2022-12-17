# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import io
import os
import pathlib
import torch
import tempfile
try:
    import tensorflow as tf
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    NO_XLA = False
except ImportError:
    NO_XLA = True

from platforms import gcp


@dataclass
class Platform(gcp.Platform):
    cores: int = 8

    _dataset_root = 'gs://jfrankle-results/open_lth/datasets'

    @property
    def device_str(self): return 'tpu'

    @property
    def torch_device(self): return xm.xla_device()

    @property
    def is_parallel(self): return True

    @property
    def is_distributed(self): return True

    @property
    def rank(self): return xm.get_ordinal()

    @property
    def world_size(self): return xm.xrt_world_size()

    def barrier(self, s='barrier'):
        xm.rendezvous(s)

    def _run_job(self, index, f):
        print(f'Starting TPU Core {self.rank}')
        self.barrier('platforms.tpu._run_job')
        super(Platform, self).run_job(f)

    def run_job(self, f):
        if NO_XLA: raise ValueError('Must install torch_xla')
        xmp.spawn(self._run_job, args=(f,), nprocs=self.cores, start_method='fork')

    def save_model(self, model, path, *args, **kwargs):
        super(Platform, self).save_model(xm._maybe_convert_to_cpu(model), path, *args, **kwargs)
