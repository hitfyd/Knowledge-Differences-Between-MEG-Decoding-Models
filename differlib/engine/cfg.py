from yacs.config import CfgNode as CN
from .utils import log_msg


def show_cfg(cfg):
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.MODEL = cfg.MODEL
    dump_cfg.AUGMENTATION = cfg.AUGMENTATION
    dump_cfg.SELECTION = cfg.SELECTION
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

# Data Augmentation
CFG.AUGMENTATION = CN()
CFG.AUGMENTATION.TYPE = "NONE"

# Feature Selection
CFG.SELECTION = CN()
CFG.SELECTION.TYPE = "NONE"
CFG.SELECTION.RATE = 0.01

# Explainer
CFG.EXPLAINER = CN()
CFG.EXPLAINER.TYPE = "LogitDeltaRule"
CFG.EXPLAINER.MAX_DEPTH = 6
CFG.EXPLAINER.MIN_SAMPLES_LEAF = 2

# Log
CFG.LOG = CN()
CFG.LOG.PREFIX = "./output"
