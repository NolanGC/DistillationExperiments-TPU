# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib

from platforms import base


class Platform(base.Platform):
    @property
    def root(self):
        import os
        dir_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(dir_path, os.pardir, os.pardir, 'open_lth_data')

    @property
    def dataset_root(self):
        return os.path.join(pathlib.Path.home(), 'open_lth_datasets')

    @property
    def imagenet_root(self):
        # return "/nobackup/projects/ILSVRC2012"
        return "/imagenet/ILSVRC2012"
        raise NotImplementedError

    @property
    def nlp_data_root(self):
        return "/working_directory"
        return "/home/tianjin/naacl_transfer_learning_tutorial"

    @property
    def iwslt_root(self):
        return self.dataset_root

    @property
    def tinyimagenet_root(self):
        return "/working_directory/tiny-imagenet-200"
        return "/nobackup/users/tianjin/tiny-imagenet-200"

    @property
    def wmt_root(self):
        raise NotImplementedError

