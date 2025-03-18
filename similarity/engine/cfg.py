from yacs.config import CfgNode as CN

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
CFG.NUM_SAMPLES = 300

# Models
CFG.MODELS = ["linear", "mlp", "hgrn", "lfcnn", "varcnn", "atcnet"]  # 按CamCAN测试集上的精度升序排列

# Explainer
CFG.EXPLAINER = CN()
CFG.EXPLAINER.TYPE = "ShapleyValueExplainer"
CFG.EXPLAINER.W = 1
CFG.EXPLAINER.M = 64
CFG.EXPLAINER.NUM_REFERENCES = 100
CFG.EXPLAINER.RANGE_REFERENCES = -1000
