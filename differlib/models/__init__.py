import os

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .DNNClassifier import lfcnn, varcnn, hgrn, mlp, linear, eegnetv4, eegnetv1
from .SoftDecisionTree import sdt
from .atcnet.atcnet import atcnet
from .atcnet_new.ctnet import ctnet
from .atcnet_new.eegnex import eegnex
from .atcnet_new.msvtnet import msvtnet
from .meegnet.network import meegnet

model_checkpoint_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../checkpoint/Models_Train/"
)

scikit_models = ["gnb", "rf", "lr"]
torch_models = ["mlp", "lfcnn", "varcnn", "hgrn", "atcnet", "linear", "sdt", "eegnetv1", "eegnetv4", "msvtnet"]

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
        "meegnet": (meegnet, model_checkpoint_prefix + "CamCAN_MEEGNet_128_0.0003_20250327111123_checkpoint.pt"),
        "eegnex": (eegnex, model_checkpoint_prefix + "CamCAN_EEGNeX_64_0.001_20250425141344_checkpoint.pt"),
        "ctnet": (ctnet, model_checkpoint_prefix + "CamCAN_CTNet_128_0.001_0.0003_0.0_20250426133227_checkpoint.pt"),
        "msvtnet": (msvtnet, model_checkpoint_prefix + "CamCAN_MSVTNet_128_0.003_0.0_1e-06_20250426133227_checkpoint.pt"),
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
        "meegnet": (meegnet, model_checkpoint_prefix + "DecMeg2014_MEEGNet_128_0.0003_20250327111123_checkpoint.pt"),
        "eegnex": (eegnex, model_checkpoint_prefix + "DecMeg2014_EEGNeX_64_0.001_20250425141344_checkpoint.pt"),
        "ctnet": (ctnet, model_checkpoint_prefix + "DecMeg2014_CTNet_64_0.003_0.0003_0.0_20250426133227_checkpoint.pt"),
        "msvtnet": (msvtnet, model_checkpoint_prefix + "DecMeg2014_MSVTNet_128_0.003_0.0_0.0_20250429100210_checkpoint.pt"),
        "sdt": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_Vanilla"),
        "sdt_hgrn_kd": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_HGRN_KD"),
        "sdt_hgrn_fakd": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_HGRN_FAKD"),
    },

    "BCIIV2a": {
        "eegnetv4": (eegnetv4, model_checkpoint_prefix + "BCIIV2a_eegnetv4"),
        "eegnetv1": (eegnetv1, model_checkpoint_prefix + "BCIIV2a_eegnetv1"),
    }
}
