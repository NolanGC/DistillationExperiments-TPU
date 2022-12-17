# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from platforms import gcp


class Platform(gcp.Platform):
    @property
    def root(self):
        return 'gs://jfrankle-results/open_lth/fairdata'
