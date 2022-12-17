# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import os
try:
    import submitit
    NO_SUBMITIT = False
except ImportError:
    NO_SUBMITIT = True

from platforms import base


if NO_SUBMITIT:
    SubmititFunc = None
else:
    class SubmititFunc(submitit.helpers.Checkpointable):
        def __init__(self, f, platform):
            self.platform = platform
            self.f = f

        def __call__(self, *args, **kwargs):
            self.platform._run_job(self.f)


@dataclass
class Platform(base.Platform):
    gpus: int = 1
    mem: int = None
    timeout_min: int = 24*60*3
    volta32: bool = False
    partition: str = 'learnfair'

    @property
    def root(self):
        return '/checkpoint/jfrankle/open_lth_data'

    @property
    def dataset_root(self):
        return '/checkpoint/jfrankle/open_lth_datasets'

    @property
    def imagenet_root(self):
        return '/datasets01_101/imagenet_full_size/061417'

    def run_job(self, f, executor=None):
        if NO_SUBMITIT: raise ValueError('Must install submitit')
        executor = executor or submitit.AutoExecutor(folder=os.path.join(self.root, 'slurm'))
        mem = self.mem or self.gpus * 64

        slargs = {
            'timeout_min': self.timeout_min,
            'partition': self.partition,
            'gpus_per_node': self.gpus,
            'cpus_per_task': self.num_workers,
        }

        if self.partition != 'learnfair':
            slargs['comment'] = 'neurips deadline 5/20 (last day at FAIR for jfrankle)'

        if self.volta32:
            slargs['constraint'] = 'volta32gb'

        slargs['mem_gb'] = mem

        executor.update_parameters(**slargs)
        job = executor.submit(SubmititFunc(f, self))

        print('SLURM JOB ID: {}'.format(job.job_id))

    def _run_job(self, f):
        super(Platform, self).run_job(f)
