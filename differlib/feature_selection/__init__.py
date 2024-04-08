from .LinearRegression import LinearRegression, SGDRegressor
from .borutaFS import BorutaFS
from .fsm import NoneFSM

fsm_dict = {
    "NONE": NoneFSM,
    "LR": LinearRegression,
    "SGDR": SGDRegressor,
    "Boruta": BorutaFS,
}
