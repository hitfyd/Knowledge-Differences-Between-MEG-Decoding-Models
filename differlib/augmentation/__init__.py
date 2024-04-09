from .am import NoneAM, BaseAM
from .smoteAM import SmoteAM

am_dict = {
    "NONE": NoneAM,
    "BASE": BaseAM,
    "SMOTE": SmoteAM,
}