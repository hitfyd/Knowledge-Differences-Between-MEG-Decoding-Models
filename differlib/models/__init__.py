import os

import cupy as cp  # 导入 CuPy
import numpy as np
import torch
from cuml import LogisticRegression
from cuml.ensemble import RandomForestClassifier
from torch import nn

from .DNNClassifier import lfcnn, varcnn, hgrn, mlp, linear, eegnetv4, eegnetv1
from .SoftDecisionTree import sdt
from .atcnet.atcnet import atcnet
from .atcnet_new.ctnet import ctnet
from .atcnet_new.eegnex import eegnex
from ..engine.utils import log_msg, load_checkpoint, predict

# from .atcnet_new.msvtnet import msvtnet
# from .meegnet.network import meegnet

model_checkpoint_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../checkpoint/Models_Train/"
)

scikit_models = ["rf", "lr"]    # cuML
torch_models = ["mlp", "lfcnn", "varcnn", "hgrn", "atcnet", "linear", "sdt", "eegnetv1", "eegnetv4", "ctnet", "eegnex"]

model_dict = {
    "CamCAN": {
        # "gnb": (GaussianNB, model_checkpoint_prefix + "CamCAN_GNB"),
        "rf": (RandomForestClassifier, model_checkpoint_prefix + "CamCAN_RF1"),
        "lr": (LogisticRegression, model_checkpoint_prefix + "CamCAN_LR"),
        "lfcnn": (lfcnn, model_checkpoint_prefix + "CamCAN_LFCNN_20220616160458_checkpoint.pt"),
        "varcnn": (varcnn, model_checkpoint_prefix + "CamCAN_VARCNN_20220616160458_checkpoint.pt"),
        "hgrn": (hgrn, model_checkpoint_prefix + "CamCAN_HGRN_20220616160458_checkpoint.pt"),
        "linear": (linear, model_checkpoint_prefix + "CamCAN_Linear_128_0.0003_20240421215048_checkpoint.pt"),
        "mlp": (mlp, model_checkpoint_prefix + "CamCAN_MLP_128_0.0003_20240421215048_checkpoint.pt"),
        "atcnet": (atcnet, model_checkpoint_prefix + "CamCAN_ATCNet_128_0.003_20240421215048_checkpoint.pt"),
        # "meegnet": (meegnet, model_checkpoint_prefix + "CamCAN_MEEGNet_128_0.0003_20250327111123_checkpoint.pt"),
        "eegnex": (eegnex, model_checkpoint_prefix + "CamCAN_EEGNeX_64_0.001_20250425141344_checkpoint.pt"),
        "ctnet": (ctnet, model_checkpoint_prefix + "CamCAN_CTNet_128_0.001_0.0003_0.0_20250426133227_checkpoint.pt"),
        # "msvtnet": (msvtnet, model_checkpoint_prefix + "CamCAN_MSVTNet_128_0.003_0.0_1e-06_20250426133227_checkpoint.pt"),
        "sdt": (sdt, model_checkpoint_prefix + "CamCAN_SDT_Vanilla"),
        "sdt_varcnn_kd": (sdt, model_checkpoint_prefix + "CamCAN_SDT_VARCNN_KD"),
        "sdt_varcnn_fakd": (sdt, model_checkpoint_prefix + "CamCAN_SDT_VARCNN_FAKD"),
    },

    "DecMeg2014": {
        # "gnb": (GaussianNB, model_checkpoint_prefix + "DecMeg2014_GNB"),
        "rf": (RandomForestClassifier, model_checkpoint_prefix + "DecMeg2014_RF1"),
        "lr": (LogisticRegression, model_checkpoint_prefix + "DecMeg2014_LR"),
        "lfcnn": (lfcnn, model_checkpoint_prefix + "DecMeg2014_LFCNN_20230601182643_checkpoint.pt"),     # "DecMeg2014_LFCNN_20220616192753_checkpoint.pt" "DecMeg2014_LFCNN_20230601182643_checkpoint.pt"
        "varcnn": (varcnn, model_checkpoint_prefix + "DecMeg2014_VARCNN_20230601184341_checkpoint.pt"),  # "DecMeg2014_VARCNN_20220616192753_checkpoint.pt" "DecMeg2014_VARCNN_20230601184341_checkpoint.pt"
        "hgrn": (hgrn, model_checkpoint_prefix + "DecMeg2014_HGRN_20220616192753_checkpoint.pt"),
        "linear": (linear, model_checkpoint_prefix + "DecMeg2014_Linear_64_0.0003_20240421215048_checkpoint.pt"),
        "mlp": (mlp, model_checkpoint_prefix + "DecMeg2014_MLP_128_0.001_20240421215048_checkpoint.pt"),
        "atcnet": (atcnet, model_checkpoint_prefix + "DecMeg2014_ATCNet_64_0.001_20240421215048_checkpoint.pt"),
        # "meegnet": (meegnet, model_checkpoint_prefix + "DecMeg2014_MEEGNet_128_0.0003_20250327111123_checkpoint.pt"),
        "eegnex": (eegnex, model_checkpoint_prefix + "DecMeg2014_EEGNeX_64_0.001_20250425141344_checkpoint.pt"),
        "ctnet": (ctnet, model_checkpoint_prefix + "DecMeg2014_CTNet_64_0.003_0.0003_0.0_20250426133227_checkpoint.pt"),
        # "msvtnet": (msvtnet, model_checkpoint_prefix + "DecMeg2014_MSVTNet_128_0.003_0.0_0.0_20250429100210_checkpoint.pt"),
        "sdt": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_Vanilla"),
        "sdt_hgrn_kd": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_HGRN_KD"),
        "sdt_hgrn_fakd": (sdt, model_checkpoint_prefix + "DecMeg2014_SDT_HGRN_FAKD"),
    },

    "BCIIV2a": {
        "eegnetv4": (eegnetv4, model_checkpoint_prefix + "BCIIV2a_eegnetv4"),
        "eegnetv1": (eegnetv1, model_checkpoint_prefix + "BCIIV2a_eegnetv1"),
    }
}


class CuMLWrapper(nn.Module):
    def __init__(self, ml_model):
        super().__init__()
        self.ml_model = ml_model
        self.__class__.__name__ = ml_model.__class__.__name__  # 替换类名为Scikit-learn类
        if isinstance(ml_model, RandomForestClassifier):
            self.ml_model = ml_model.convert_to_fil_model(output_class=True)   # 重点在于转化为推理模型，速度大幅加快

    def forward(self, x):
        x = x.flatten(1)
        x_cupy = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))

        # 获取预测结果（根据需求选择预测方法）
        if hasattr(self.ml_model, "predict_proba"):
            y_pred = self.ml_model.predict_proba(x_cupy)
        else:
            y_pred = self.ml_model.predict(x_cupy)

        # 转回PyTorch Tensor
        output= torch.as_tensor(y_pred, device=x.device)
        return output


def load_pretrained_model(model_type, dataset, channels, points, n_classes, device: torch.device = torch.device("cpu")):
    print(log_msg("Loading model {}".format(model_type), "INFO"))
    model_class, model_pretrain_path = model_dict[dataset][model_type]
    assert (model_pretrain_path is not None), "no pretrain model {}".format(model_type)
    pretrained_model = None
    if model_type in scikit_models:
        pretrained_model = load_checkpoint(model_pretrain_path, device)
        pretrained_model = CuMLWrapper(pretrained_model).to(device)
    elif model_type in torch_models:
        pretrained_model = model_class(channels=channels, points=points, num_classes=n_classes)
        pretrained_model.load_state_dict(load_checkpoint(model_pretrain_path, device))
        pretrained_model = pretrained_model.to(device)
    else:
        print(log_msg("No pretrain model {} found".format(model_type), "INFO"))
    assert pretrained_model is not None
    return pretrained_model


def output_predict_targets(model_type, model, data: np.ndarray, num_classes=2, batch_size=512, softmax=True):
    output, predict_targets = None, None
    if model_type in scikit_models:
        output = predict(model, data, num_classes=num_classes, batch_size=batch_size, softmax=softmax, eval=True)
        predict_targets = np.argmax(output, axis=1)
    elif model_type in torch_models:
        output = predict(model, data, num_classes=num_classes, batch_size=batch_size, softmax=softmax, eval=True)
        predict_targets = np.argmax(output, axis=1)
    else:
        print(log_msg("No pretrain model {} found".format(model_type), "INFO"))
    assert output is not None
    assert predict_targets is not None
    return output, predict_targets