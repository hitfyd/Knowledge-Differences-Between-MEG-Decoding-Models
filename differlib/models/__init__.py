import os

from .DNNClassifier import lfcnn, varcnn, hgrn
from .SoftDecisionTree import sdt


model_checkpoint_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../checkpoint/Models_Train/"
)

model_dict = {
    # teachers
    "CamCAN_lfcnn": (lfcnn, model_checkpoint_prefix + "CamCAN_LFCNN_20220616160458_checkpoint.pt"),
    "CamCAN_varcnn": (varcnn, model_checkpoint_prefix + "CamCAN_VARCNN_20220616160458_checkpoint.pt"),
    "CamCAN_hgrn": (hgrn, model_checkpoint_prefix + "CamCAN_HGRN_20220616160458_checkpoint.pt"),
    "CamCAN_sdt": (sdt, model_checkpoint_prefix + "CamCAN_SDT_Vanilla"),
    "CamCAN_sdt3": (sdt, model_checkpoint_prefix + "CamCAN_SDT3_Vanilla"),
    "CamCAN_sdt4": (sdt, model_checkpoint_prefix + "CamCAN_SDT4_Vanilla"),
    "CamCAN_sdt_varcnn_kd": (sdt, model_checkpoint_prefix + "CamCAN_SDT_VARCNN_KD"),
    "CamCAN_sdt_varcnn_fakd": (sdt, model_checkpoint_prefix + "CamCAN_SDT_VARCNN_FAKD"),
    "CamCAN_sdt3_varcnn_fakd": (sdt, model_checkpoint_prefix + "CamCAN_SDT3_VARCNN_FAKD"),
    "CamCAN_sdt4_varcnn_fakd": (sdt, model_checkpoint_prefix + "CamCAN_SDT4_VARCNN_FAKD"),

    "DecMeg2014_lfcnn": (lfcnn, model_checkpoint_prefix + "DecMeg2014_LFCNN_20230601182643_checkpoint.pt"),     # "DecMeg2014_LFCNN_20220616192753_checkpoint.pt" "DecMeg2014_LFCNN_20230601182643_checkpoint.pt"
    "DecMeg2014_varcnn": (varcnn, model_checkpoint_prefix + "DecMeg2014_VARCNN_20230601184341_checkpoint.pt"),  # "DecMeg2014_VARCNN_20220616192753_checkpoint.pt" "DecMeg2014_VARCNN_20230601184341_checkpoint.pt"
    "DecMeg2014_hgrn": (hgrn, model_checkpoint_prefix + "DecMeg2014_HGRN_20220616192753_checkpoint.pt"),
    "DecMeg2014_sdt": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_Vanilla"),
    "DecMeg2014_sdt_hgrn_kd": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_HGRN_KD"),
    "DecMeg2014_sdt_hgrn_fakd": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_HGRN_FAKD"),


    # students
    "sdt": (sdt, None),
}
