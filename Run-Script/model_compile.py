import os
import time

import torch

from differlib.engine.utils import save_checkpoint, dataset_info_dict, model_eval, get_data_labels_from_dataset, \
    get_data_loader, load_checkpoint
from differlib.models import model_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
save_path = "./compile_models/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# init dataset & models
for dataset in ["CamCAN", "DecMeg2014"]:
    channels, points, n_classes = dataset_info_dict[dataset].values()
    test_data, test_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    test_loader = get_data_loader(test_data, test_labels)

    for model_type in ['atcnet']:    # 只有atcnet有正收益
        model_class, model_pretrain_path = model_dict[dataset][model_type]
        model = model_class(channels=channels, points=points, num_classes=n_classes)
        model.load_state_dict(load_checkpoint(model_pretrain_path, device))
        time_start = time.perf_counter()
        original_accuracy = model_eval(model, test_loader)
        original_run_time = time.perf_counter() - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

        cache_file = os.path.join(save_path, f"{dataset}_{model_type}_compile_cache.pkl")
        if os.path.exists(cache_file):
            artifact_bytes = torch.load(cache_file)
            torch.compiler.load_cache_artifacts(artifact_bytes)

        compile_model = torch.compile(model, mode="max-autotune", dynamic=True)    # reduce-overhead    max-autotune
        compile_accuracy = model_eval(compile_model, test_loader)  # 第一次编译

        time_start_ = time.perf_counter()
        compile_accuracy = model_eval(compile_model, test_loader)
        compile_run_time = time.perf_counter() - time_start_

        print(
            f"Dataset {dataset} Model {model_type} accuracy is {original_accuracy} {original_run_time}s, Compile accuracy: {compile_accuracy} {compile_run_time}s")

        if not os.path.exists(cache_file):
            artifacts = torch.compiler.save_cache_artifacts()
            assert artifacts is not None
            artifact_bytes, cache_info = artifacts
            torch.save(artifact_bytes, cache_file)