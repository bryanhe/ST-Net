import os
import configparser
import types

FILENAME = None
param = {}
for filename in ["stnet.cfg",
                 ".stnet.cfg",
                 os.path.expanduser("~/stnet.cfg"),
                 os.path.expanduser("~/.stnet.cfg")]:
    if os.path.isfile(filename):
        FILENAME = filename
        config = configparser.ConfigParser()
        with open(filename, "r") as f:
            config.read_string("[config]\n" + f.read())
            param = config["config"]
        break

config = types.SimpleNamespace(FILENAME = FILENAME,
                               SPATIAL_RAW_ROOT = param.get("spatial_raw_root", "data/hist2tscript/"),
                               SPATIAL_PROCESSED_ROOT = param.get("spatial_processed_root","data/hist2tscript-patch/"),
                               TCGA_RAW_ROOT = param.get("tcga_raw_root", "data/TCGA/"),
                               TCGA_PROCESSED_ROOT = param.get("tcga_processed_root","data/TCGA-patch/"))
