import os

import cupy as cp  # 导入 CuPy
import torch
from cuml import LogisticRegression
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score

from differlib.engine.utils import save_checkpoint, load_checkpoint, get_data_labels_from_dataset, setup_seed

random_state = 1234
setup_seed(random_state)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# datasets
datasets = [
    "CamCAN",
    "DecMeg2014"
]
models = dict(LR=LogisticRegression(random_state=random_state),
              RF1=RandomForestClassifier(random_state=random_state, n_streams=1),)

# log config
log_path = f"./output/MEG_cuML/"
if not os.path.exists(log_path):
    os.makedirs(log_path)

# init dataset & models
for dataset in datasets:
    x_train, y_train = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
    x_test_all, y_test_all = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    _, channels, points = x_train.shape
    num_classes = len(set(y_train))
    x_train, y_train = torch.from_numpy(x_train).to(device), torch.from_numpy(y_train).to(device)
    x_test_all, y_test_all = torch.from_numpy(x_test_all).to(device), torch.from_numpy(y_test_all).to(device)
    x_train = x_train.flatten(1)
    x_test_all = x_test_all.flatten(1)
    x_train, y_train = cp.asarray(x_train), cp.asarray(y_train)
    x_test_all, y_test_all = cp.asarray(x_test_all), cp.asarray(y_test_all)

    # Training models and save checkpoints
    for model_name, model in models.items():
        save_path = os.path.join(log_path, "{}_{}".format(dataset, model_name))
        if not os.path.exists(save_path):
            model.fit(x_train, y_train)
            save_checkpoint(model, save_path)
        else:
            model = load_checkpoint(save_path, device)

        if isinstance(model, RandomForestClassifier):
            model = model.convert_to_fil_model(output_class=True)

        acc = accuracy_score(y_true=y_train, y_pred=model.predict(x_train))
        t_acc = accuracy_score(y_true=y_test_all, y_pred=model.predict(x_test_all))
        print(f"dataset: {dataset} model: {model_name} train accuracy: {(acc * 100):.4f}% test accuracy: {(t_acc * 100):.4f}%")
        with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
            writer.write(f"dataset: {dataset} model: {model_name} train accuracy: {(acc * 100):.4f}% test accuracy: {(t_acc * 100):.4f}%" + os.linesep)
