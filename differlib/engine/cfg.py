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
CFG.EXPERIMENT.NUM_REPETITIONS = 5
CFG.EXPERIMENT.GPU_IDS = "1"    # List of GPUs used
CFG.EXPERIMENT.CPU_COUNT = 8

# Dataset
CFG.DATASET = CN()
CFG.DATASET.TYPE = "CamCAN"
CFG.DATASET.CHANNELS = 204
CFG.DATASET.POINTS = 100
CFG.DATASET.NUM_CLASSES = 2
CFG.DATASET.NUM_SPLITS = 5

# Models
CFG.MODELS = CN()
CFG.MODELS.A = "CamCAN_varcnn"
CFG.MODELS.B = "CamCAN_sdt"

# Data Augmentation
CFG.AUGMENTATION = CN()
CFG.AUGMENTATION.TYPE = "NONE"

# Normalization
CFG.NORMALIZATION = CN()
CFG.NORMALIZATION.FLAG = True

# Feature Extraction
CFG.EXTRACTION = CN()
CFG.EXTRACTION.FLAG = True
CFG.EXTRACTION.WINDOW_LENGTH = 10

# Feature Selection
CFG.SELECTION = CN()
CFG.SELECTION.TYPE = "NONE"
CFG.SELECTION.RATE = 0.01
CFG.SELECTION.Diff = CN()
CFG.SELECTION.Diff.WINDOW_LENGTH = 10
CFG.SELECTION.Diff.M = 8

# Explainer
CFG.EXPLAINER = CN()
CFG.EXPLAINER.TYPE = "LogitDeltaRule"
CFG.EXPLAINER.MAX_DEPTH = 5
CFG.EXPLAINER.MIN_SAMPLES_LEAF = 1

# Log
CFG.LOG = CN()
CFG.LOG.PREFIX = "./output"
