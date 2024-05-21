from .MERLINXAI import MERLINXAI
from .imd.imd import SeparateSurrogate, IMDExplainer
from .DeltaXplainer import DeltaExplainer
from .LogitDeltaRule import LogitDeltaRule

explainer_dict = {
    "SS": SeparateSurrogate,
    "IMD": IMDExplainer,
    "Delta": DeltaExplainer,
    "Logit": LogitDeltaRule,
    "MERLIN": MERLINXAI,
}
