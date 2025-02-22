import os

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .DNNClassifier import lfcnn, varcnn, hgrn, mlp, linear
from .SoftDecisionTree import sdt
from .atcnet.atcnet import atcnet

model_checkpoint_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../checkpoint/Models_Train/"
)

scikit_models = ["gnb", "rf", "lr"]
torch_models = ["mlp", "lfcnn", "varcnn", "hgrn", "atcnet", "linear", "sdt"]

model_dict = {
    "CamCAN": {
        "gnb": (GaussianNB, model_checkpoint_prefix + "CamCAN_GNB"),
        "rf": (RandomForestClassifier, model_checkpoint_prefix + "CamCAN_RF1"),
        "lr": (LogisticRegression, model_checkpoint_prefix + "CamCAN_LR"),
        "lfcnn": (lfcnn, model_checkpoint_prefix + "CamCAN_LFCNN_20220616160458_checkpoint.pt"),
        "varcnn": (varcnn, model_checkpoint_prefix + "CamCAN_VARCNN_20220616160458_checkpoint.pt"),
        "hgrn": (hgrn, model_checkpoint_prefix + "CamCAN_HGRN_20220616160458_checkpoint.pt"),
        "linear": (linear, model_checkpoint_prefix + "CamCAN_Linear_128_0.0003_20240421215048_checkpoint.pt"),
        "mlp": (mlp, model_checkpoint_prefix + "CamCAN_MLP_128_0.0003_20240421215048_checkpoint.pt"),
        "atcnet": (atcnet, model_checkpoint_prefix + "CamCAN_ATCNet_128_0.003_20240421215048_checkpoint.pt"),
        "sdt": (sdt, model_checkpoint_prefix + "CamCAN_SDT_Vanilla"),
        "sdt_varcnn_kd": (sdt, model_checkpoint_prefix + "CamCAN_SDT_VARCNN_KD"),
        "sdt_varcnn_fakd": (sdt, model_checkpoint_prefix + "CamCAN_SDT_VARCNN_FAKD"),
    },

    "DecMeg2014": {
        "gnb": (GaussianNB, model_checkpoint_prefix + "DecMeg2014_GNB"),
        "rf": (RandomForestClassifier, model_checkpoint_prefix + "DecMeg2014_RF1"),
        "lr": (LogisticRegression, model_checkpoint_prefix + "DecMeg2014_LR"),
        "lfcnn": (lfcnn, model_checkpoint_prefix + "DecMeg2014_LFCNN_20230601182643_checkpoint.pt"),     # "DecMeg2014_LFCNN_20220616192753_checkpoint.pt" "DecMeg2014_LFCNN_20230601182643_checkpoint.pt"
        "varcnn": (varcnn, model_checkpoint_prefix + "DecMeg2014_VARCNN_20230601184341_checkpoint.pt"),  # "DecMeg2014_VARCNN_20220616192753_checkpoint.pt" "DecMeg2014_VARCNN_20230601184341_checkpoint.pt"
        "hgrn": (hgrn, model_checkpoint_prefix + "DecMeg2014_HGRN_20220616192753_checkpoint.pt"),
        "linear": (linear, model_checkpoint_prefix + "DecMeg2014_Linear_64_0.0003_20240421215048_checkpoint.pt"),
        "mlp": (mlp, model_checkpoint_prefix + "DecMeg2014_MLP_128_0.001_20240421215048_checkpoint.pt"),
        "atcnet": (atcnet, model_checkpoint_prefix + "DecMeg2014_ATCNet_64_0.001_20240421215048_checkpoint.pt"),
        "sdt": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_Vanilla"),
        "sdt_hgrn_kd": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_HGRN_KD"),
        "sdt_hgrn_fakd": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_HGRN_FAKD"),
    },
}
