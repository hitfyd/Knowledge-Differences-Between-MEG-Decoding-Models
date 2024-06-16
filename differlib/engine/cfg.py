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
CFG.EXPERIMENT.PROJECT = ""
CFG.EXPERIMENT.NAME = ""
CFG.EXPERIMENT.TAG = ""
CFG.EXPERIMENT.SEED = 0  # Random number seed, which is beneficial to the repeatability of the experiment. 0/2024
CFG.EXPERIMENT.GPU_IDS = "1"    # List of GPUs used
CFG.EXPERIMENT.CPU_COUNT = 3

# Dataset
CFG.DATASET = "CamCAN"
CFG.NUM_SPLITS = 5
CFG.WINDOW_LENGTH = 5

# Models
CFG.MODEL_A = "mlp"
CFG.MODEL_B = "atcnet"

# Data Augmentation
CFG.AUGMENTATION = "NONE"
CFG.AUGMENT_FACTOR = 1.0

# Feature Selection
CFG.SELECTION = CN()
CFG.SELECTION.TYPE = "NONE"
CFG.SELECTION.Diff = CN()
CFG.SELECTION.Diff.M = 8
CFG.SELECTION.Diff.THRESHOLD = 6.0

# Explainer
CFG.EXPLAINER = CN()
CFG.EXPLAINER.TYPE = "Logit"
CFG.EXPLAINER.MAX_DEPTH = 7
CFG.EXPLAINER.MIN_SAMPLES_LEAF = 1

# Log
CFG.LOG = CN()
CFG.LOG.PREFIX = "./output"
