from .DiffShapleyFS import DiffShapleyFS
from .fsm import NoneFSM

fsm_dict = {
    "NONE": NoneFSM,
    "DiffShapley": DiffShapleyFS,
}
