from platforms import gcp, platform
platform._PLATFORM = gcp.Platform()
from platforms.platform import get_platform
import os

root_loc = "gs://tianjin-distgen/nolan/"

def save_to_gcp(path, payload):
    with get_platform().open(os.path.join(root_loc, path), "wb") as fp:
        fp.write(payload)

