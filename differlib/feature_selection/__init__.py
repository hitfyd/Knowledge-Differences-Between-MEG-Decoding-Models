from .LinearRegression import LinearRegression, SGDRegressor
from .fsm import NoneFSM

fsm_dict = {
    "None": NoneFSM,
    "LinearRegression": LinearRegression,
    "SGDRegressor": SGDRegressor,
}
