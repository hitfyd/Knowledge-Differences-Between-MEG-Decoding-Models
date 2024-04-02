from .imd.imd import SeparateSurrogate, IMDExplainer
from .DeltaXplainer import DeltaExplainer
from .LogitDeltaRule import LogitDeltaRule

explainer_dict = {
    "SeparateSurrogate": SeparateSurrogate(),
    "IMDExplainer": IMDExplainer(),
    "DeltaExplainer": DeltaExplainer(),
    "LogitDeltaRule": LogitDeltaRule(),
}
