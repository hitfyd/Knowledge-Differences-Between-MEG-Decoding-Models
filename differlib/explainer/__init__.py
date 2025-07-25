from .DeltaXplainer_official import DeltaExplainer_Official
from .MERLINXAI import MERLINXAI
from .imd.imd import SeparateSurrogate, IMDExplainer
from .DeltaXplainer import DeltaExplainer
from .LogitDeltaRule import LogitDeltaRule

explainer_dict = {
    "SS": SeparateSurrogate,
    "IMD": IMDExplainer,
    "Delta": DeltaExplainer,
    "Delta_Official": DeltaExplainer_Official,
    "Logit": LogitDeltaRule,
    "MERLIN": MERLINXAI,
}
