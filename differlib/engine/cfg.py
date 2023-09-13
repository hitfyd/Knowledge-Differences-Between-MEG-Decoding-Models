from yacs.config import CfgNode as CN
from .utils import log_msg


def show_cfg(cfg):
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.DISTILLER = cfg.DISTILLER
    dump_cfg.SOLVER = cfg.SOLVER
    dump_cfg.LOG = cfg.LOG
    if cfg.DISTILLER.TYPE in cfg:
        dump_cfg.update({cfg.DISTILLER.TYPE: cfg.get(cfg.DISTILLER.TYPE)})
    print(log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))


CFG = CN()

# Experiment
CFG.EXPERIMENT = CN()
CFG.EXPERIMENT.PROJECT = "distill"
CFG.EXPERIMENT.NAME = ""
CFG.EXPERIMENT.TAG = "default"
CFG.EXPERIMENT.SEED = 0  # Random number seed, which is beneficial to the repeatability of the experiment.
CFG.EXPERIMENT.GPU_IDS = "0, 1"    # List of GPUs used
CFG.EXPERIMENT.REPETITION_NUM = 5   # Number of repetition times

# Dataset
CFG.DATASET = CN()
CFG.DATASET.TYPE = "CamCAN"
CFG.DATASET.CHANNELS = 204
CFG.DATASET.POINTS = 100
CFG.DATASET.NUM_CLASSES = 2
CFG.DATASET.NUM_WORKERS = 2
CFG.DATASET.TEST = CN()
CFG.DATASET.TEST.BATCH_SIZE = 1024

# Solver
CFG.SOLVER = CN()
CFG.SOLVER.TRAINER = "base"
CFG.SOLVER.BATCH_SIZE = 1024   # Grid search
CFG.SOLVER.EPOCHS = 100
CFG.SOLVER.LR = 0.003
# CFG.SOLVER.LR_DECAY_STAGES = [150, 180, 210]
# CFG.SOLVER.LR_DECAY_RATE = 0.1
CFG.SOLVER.WEIGHT_DECAY = 0.0005
CFG.SOLVER.MOMENTUM = 0.9
CFG.SOLVER.TYPE = "SGD"

# Models
CFG.MODELS = CN()
CFG.MODELS.A = "CamCAN_sdt_varcnn_fakd"
CFG.MODELS.B = "CamCAN_sdt"

# GLOBAL
CFG.GLOBAL = CN()

# LOCAL
CFG.LOCAL = CN()

# Log
CFG.LOG = CN()
# CFG.LOG.SAVE_CHECKPOINT_FREQ = 20
CFG.LOG.PREFIX = "./output"
CFG.LOG.WANDB = False

# Distillation Methods
