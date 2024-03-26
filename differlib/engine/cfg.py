from yacs.config import CfgNode as CN
from .utils import log_msg


def show_cfg(cfg):
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.MODEL = cfg.MODEL
    dump_cfg.EXPLAINER = cfg.EXPLAINER
    dump_cfg.LOG = cfg.LOG
    print(log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))


CFG = CN()

# Experiment
CFG.EXPERIMENT = CN()
CFG.EXPERIMENT.PROJECT = "distill"
CFG.EXPERIMENT.NAME = ""
CFG.EXPERIMENT.TAG = "default"
CFG.EXPERIMENT.SEED = 0  # Random number seed, which is beneficial to the repeatability of the experiment.
CFG.EXPERIMENT.GPU_IDS = "0, 1"    # List of GPUs used

# Dataset
CFG.DATASET = CN()
CFG.DATASET.TYPE = "CamCAN"
CFG.DATASET.CHANNELS = 204
CFG.DATASET.POINTS = 100
CFG.DATASET.NUM_CLASSES = 2
CFG.DATASET.NUM_SPLITS = 3

# Models
CFG.MODELS = CN()
CFG.MODELS.A = "CamCAN_varcnn"
CFG.MODELS.B = "CamCAN_sdt"

# Explainer
CFG.EXPLAINER = CN()
CFG.EXPLAINER.TYPE = "LogitDeltaRule"
CFG.EXPLAINER.MAX_DEPTH = 7

# Log
CFG.LOG = CN()
CFG.LOG.PREFIX = "./output"
