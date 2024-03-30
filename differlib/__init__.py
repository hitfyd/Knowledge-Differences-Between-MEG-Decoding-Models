from differlib.DeltaXpainer import DeltaExplainer
from differlib.LogitDeltaRule import LogitDeltaRule
from differlib.LogitDeltaRule.Regression import Regression
from differlib.imd.imd import SeparateSurrogate, IMDExplainer

explainer_dict = {
    "Regression": Regression(),
    "SeparateSurrogate": SeparateSurrogate(),
    "IMDExplainer": IMDExplainer(),
    "DeltaExplainer": DeltaExplainer(),
    "LogitDeltaRule": LogitDeltaRule(),
}