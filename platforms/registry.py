# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from platforms import local, gcp, tpu, gcp_fair, faircluster


registered_platforms = {
    'local': local.Platform,
    'gcp': gcp.Platform,
    'gcp_fair': gcp_fair.Platform,
    'tpu': tpu.Platform,
    'faircluster': faircluster.Platform,
}


def get(name):
    return registered_platforms[name]
