# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pathlib
import torch
try:
    import tensorflow as tf
except ImportError:
    pass
import tempfile

from platforms import base
from platforms.platform import get_platform


class Platform(base.Platform):

    _dataset_root = 'gs://jfrankle-results/open_lth/datasets'

    @property
    def root(self):
        return 'gs://tianjin-openlth-data/open_lth/data'

    @property
    def dataset_root(self):
        return Platform._dataset_root

    @property
    def imagenet_root(self):
        return '/mnt/disks/imagenet-disk/imagenet'

    @property
    def tinyimagenet_root(self):
        return '/mnt/disks/imagenet-disk/tinyimagenet'

    @property
    def wmt_root(self):
        return self.dataset_root()

    @property
    def iwslt_root(self):
        return '/mnt/disks/imagenet-disk/iwslt_de_en'

    @property
    def nlp_data_root(self):
        raise NotImplementedError
        return

    @staticmethod
    def open(file, mode='r'): return tf.io.gfile.GFile(file, mode)

    @staticmethod
    def exists(file): return tf.io.gfile.exists(file)

    @staticmethod
    def isdir(path): return tf.io.gfile.isdir(path)

    @staticmethod
    def listdir(path): return tf.io.gfile.listdir(path)

    @staticmethod
    def makedirs(path): return tf.io.gfile.makedirs(path)

    @staticmethod
    def rmtree(path):
        if tf.io.gfile.isdir(path):
            return tf.io.gfile.rmtree(path)
        else:
            return tf.io.gfile.remove(path)

    @staticmethod
    def copyfile(path1, path2):
        tf.io.gfile.copy(path1, path2)

    @classmethod
    def copytree(cls, source, dest):
        for source_subdir, subdirnames, filenames in tf.io.gfile.walk(source):
            relpath = os.path.relpath(source_subdir, source)
            dest_subdir = dest if relpath == '.' else os.path.join(dest, relpath)
            for filename in filenames:
                if not cls.exists(dest_subdir): cls.makedirs(dest_subdir)
                if cls.exists(os.path.join(dest_subdir, filename)): continue
                tf.io.gfile.copy(os.path.join(source_subdir, filename), os.path.join(dest_subdir, filename))


    @staticmethod
    def _clean():
        dir = '/tmp'
        prefixes = ['gcp-transfer', 'tmp_file_tensorflow']
        for f in os.listdir(dir):
            if any([f.startswith(prefix) for prefix in prefixes]):
                Platform.rmtree(os.path.join(dir, f))

    @staticmethod
    def save_model(model, path, *args, **kwargs):
        Platform._clean()

        dir, prefix = '.tmp', 'gcp-transfer'
        if not Platform.exists(dir): Platform.makedirs(dir)
        tmp = tempfile.mkstemp(dir=dir, prefix=prefix)[1]
        torch.save(model, tmp, *args, **kwargs)
        tf.io.gfile.copy(tmp, path, overwrite=True)

        Platform.rmtree(tmp)

    @staticmethod
    def load_model(path, primary_process_only=False, *args, **kwargs):
        dir, prefix = '.tmp', 'checkpoint'
        # If not in distributed mode, simply use temporary directory.
        if get_platform().world_size == 1:
            temp_dir = tempfile.TemporaryDirectory()
            tf.io.gfile.copy(path, os.path.join(temp_dir.name, prefix), overwrite=True)
            m = torch.load(os.path.join(temp_dir.name, prefix), *args, **kwargs)
            # use temp_dir, and when done:
            temp_dir.cleanup()
            return m

        if get_platform().is_primary_process:
            if Platform.exists(dir): Platform.rmtree(dir)
            Platform.makedirs(dir)
            tf.io.gfile.copy(path, os.path.join(dir, prefix), overwrite=True)
        if not primary_process_only: get_platform().barrier('platforms.gcp.load_model.1')

        m = torch.load(os.path.join(dir, prefix), *args, **kwargs)
        if not primary_process_only: get_platform().barrier('platforms.gcp.load_model.2')

        return m
