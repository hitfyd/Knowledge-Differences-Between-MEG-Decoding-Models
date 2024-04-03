from .LinearRegression import LinearRegression, SGDRegressor
from .fsm import BaseFSM

fsm_dict = {
    "None": BaseFSM(),
    "LinearRegression": LinearRegression(),
    "SGDRegressor": SGDRegressor(),
}
