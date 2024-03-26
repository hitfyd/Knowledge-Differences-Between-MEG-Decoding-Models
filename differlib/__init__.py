from differlib.DeltaXpainer import DeltaExplainer
from differlib.LogitDeltaRule import LogitDeltaRule
from differlib.imd.imd import SeparateSurrogate, IMDExplainer

explainer_dict = {
    "SeparateSurrogate": SeparateSurrogate(),
    "IMDExplainer": IMDExplainer(),
    "DeltaExplainer": DeltaExplainer(),
    "LogitDeltaRule": LogitDeltaRule(),
}