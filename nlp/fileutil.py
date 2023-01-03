import io
import os
import pathlib
import torch

try:
    import tensorflow as tf
except ImportError:
    pass
import tempfile

import torch_xla.core.xla_model as xm

class Platform:

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
        if xm.is_master_ordinal():
            if Platform.exists(dir): Platform.rmtree(dir)
            Platform.makedirs(dir)
            tf.io.gfile.copy(path, os.path.join(dir, prefix), overwrite=True)
        if not primary_process_only: xm.barrier('platforms.gcp.load_model.1')

        m = torch.load(os.path.join(dir, prefix), *args, **kwargs)
        if not primary_process_only: xm.barrier('platforms.gcp.load_model.2')

        return m