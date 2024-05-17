from .DiffShapleyFS import DiffShapleyFS
from .LinearRegression import LinearRegression, SGDRegressor
from .borutaFS import BorutaFS
from .fsm import NoneFSM, RandFSM

fsm_dict = {
    "NONE": NoneFSM,
    "Rand": RandFSM,
    "LR": LinearRegression,
    "SGDR": SGDRegressor,
    "Boruta": BorutaFS,
    "DiffShapley": DiffShapleyFS,
}
