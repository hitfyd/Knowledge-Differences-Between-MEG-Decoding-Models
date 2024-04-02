from .LinearRegression import LinearRegression, SGDRegressor
from .fsm import BaseFSM

fsm_dict = {
    "Base": BaseFSM(),
    "LinearRegression": LinearRegression(),
    "SGDRegressor": SGDRegressor(),
}
