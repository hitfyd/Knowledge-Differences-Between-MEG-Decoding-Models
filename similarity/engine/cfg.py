from yacs.config import CfgNode as CN
from .utils import log_msg


def show_cfg(cfg):
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.MODELS = cfg.MODELS
    dump_cfg.EXPLAINER = cfg.EXPLAINER
    dump_cfg.LOG = cfg.LOG
    print(log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))


CFG = CN()

# Experiment
CFG.EXPERIMENT = CN()
CFG.EXPERIMENT.PROJECT = ""
CFG.EXPERIMENT.NAME = ""
CFG.EXPERIMENT.TAG = ""
CFG.EXPERIMENT.SEED = 0  # Random number seed, which is beneficial to the repeatability of the experiment. 0/2024
CFG.EXPERIMENT.GPU_IDS = "0"    # List of GPUs used

# Log
CFG.LOG = CN()
CFG.LOG.PREFIX = "./output"

# Dataset
CFG.DATASET = "CamCAN"  # "DecMeg2014"

# Models
CFG.MODELS = ["mlp", "lfcnn", "varcnn", "hgrn", "atcnet", "linear", "sdt"]

# Explainer
CFG.EXPLAINER = CN()
CFG.EXPLAINER.TYPE = "ShapleyValueExplainer"
CFG.EXPLAINER.W = 1
CFG.EXPLAINER.M = 64
