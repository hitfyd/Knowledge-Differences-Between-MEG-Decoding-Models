import os

from .DNNClassifier import lfcnn, varcnn, hgrn, mlp, linear
from .SoftDecisionTree import sdt
from .atcnet.atcnet import atcnet

model_checkpoint_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../checkpoint/Models_Train/"
)

model_dict = {
    # teachers
    "CamCAN_lfcnn": (lfcnn, model_checkpoint_prefix + "CamCAN_LFCNN_20220616160458_checkpoint.pt"),
    "CamCAN_varcnn": (varcnn, model_checkpoint_prefix + "CamCAN_VARCNN_20220616160458_checkpoint.pt"),
    "CamCAN_hgrn": (hgrn, model_checkpoint_prefix + "CamCAN_HGRN_20220616160458_checkpoint.pt"),
    "CamCAN_linear": (linear, model_checkpoint_prefix + "CamCAN_Linear_128_0.0003_20240421215048_checkpoint.pt"),
    "CamCAN_mlp": (mlp, model_checkpoint_prefix + "CamCAN_MLP_128_0.0003_20240421215048_checkpoint.pt"),
    "CamCAN_atcnet": (atcnet, model_checkpoint_prefix + "CamCAN_ATCNet_128_0.003_20240421215048_checkpoint.pt"),
    "CamCAN_sdt": (sdt, model_checkpoint_prefix + "CamCAN_SDT_Vanilla"),
    "CamCAN_sdt_varcnn_kd": (sdt, model_checkpoint_prefix + "CamCAN_SDT_VARCNN_KD"),
    "CamCAN_sdt_varcnn_fakd": (sdt, model_checkpoint_prefix + "CamCAN_SDT_VARCNN_FAKD"),

    "DecMeg2014_lfcnn": (lfcnn, model_checkpoint_prefix + "DecMeg2014_LFCNN_20230601182643_checkpoint.pt"),     # "DecMeg2014_LFCNN_20220616192753_checkpoint.pt" "DecMeg2014_LFCNN_20230601182643_checkpoint.pt"
    "DecMeg2014_varcnn": (varcnn, model_checkpoint_prefix + "DecMeg2014_VARCNN_20230601184341_checkpoint.pt"),  # "DecMeg2014_VARCNN_20220616192753_checkpoint.pt" "DecMeg2014_VARCNN_20230601184341_checkpoint.pt"
    "DecMeg2014_hgrn": (hgrn, model_checkpoint_prefix + "DecMeg2014_HGRN_20220616192753_checkpoint.pt"),
    "DecMeg2014_linear": (linear, model_checkpoint_prefix + "DecMeg2014_Linear_64_0.0003_20240421215048_checkpoint.pt"),
    "DecMeg2014_mlp": (mlp, model_checkpoint_prefix + "DecMeg2014_MLP_128_0.001_20240421215048_checkpoint.pt"),
    "DecMeg2014_atcnet": (atcnet, model_checkpoint_prefix + "DecMeg2014_ATCNet_64_0.001_20240421215048_checkpoint.pt"),
    "DecMeg2014_sdt": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_Vanilla"),
    "DecMeg2014_sdt_hgrn_kd": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_HGRN_KD"),
    "DecMeg2014_sdt_hgrn_fakd": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_HGRN_FAKD"),


    # students
    "sdt": (sdt, None),
}
