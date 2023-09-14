import os
import argparse

import graphviz
import torch
import torch.nn as nn
import torchsummary
from torchviz import make_dot

from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import log_msg, setup_seed, get_data_loader_from_dataset, load_checkpoint, accuracy
from differlib.models import model_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS
    # set the random number seed
    setup_seed(cfg.EXPERIMENT.SEED)

    # init dataloader & models
    # train_loader = get_data_loader_from_dataset('../dataset/{}_train.npz'.format(cfg.DATASET.TYPE),
    #                                             cfg.SOLVER.BATCH_SIZE)
    val_loader = get_data_loader_from_dataset('../dataset/{}_test.npz'.format(cfg.DATASET.TYPE),
                                              cfg.DATASET.TEST.BATCH_SIZE)

    print(log_msg("Loading teacher model", "INFO"))
    model_A_type, model_A_pretrain_path = model_dict[cfg.MODELS.A]
    assert (model_A_pretrain_path is not None), "no pretrain model for teacher {}".format(cfg.DISTILLER.A)
    model_A = model_A_type(
        channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES)
    model_A.load_state_dict(load_checkpoint(model_A_pretrain_path))
    model_A = model_A.cuda()

    model_B_type, model_B_pretrain_path = model_dict[cfg.MODELS.B]
    assert (model_B_pretrain_path is not None), "no pretrain model for teacher {}".format(cfg.DISTILLER.B)
    model_B = model_B_type(
        channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES)
    model_B.load_state_dict(load_checkpoint(model_B_pretrain_path))
    model_B = model_B.cuda()

    # torchsummary.summary(model_A, (cfg.DATASET.CHANNELS, cfg.DATASET.POINTS))
    #
    # out_A = model_A(torch.randn(cfg.SOLVER.BATCH_SIZE, cfg.DATASET.CHANNELS, cfg.DATASET.POINTS).cuda())
    # g = make_dot(out_A, params=dict(model_A.named_parameters()), show_attrs=True, show_saved=True)  # 实例化 make_dot
    # g.view(filename="out_A")  # 直接在当前路径下保存 pdf 并打开
    #
    # out_B = model_B(torch.randn(cfg.SOLVER.BATCH_SIZE, cfg.DATASET.CHANNELS, cfg.DATASET.POINTS).cuda())
    # g = make_dot(out_B, params=dict(model_A.named_parameters()), show_attrs=True, show_saved=True)  # 实例化 make_dot
    # g.view(filename="out_B")  # 直接在当前路径下保存 pdf 并打开
    #
    # input_names = ["MEG"]
    # output_names = ["MEG Prediction"]
    #
    # torch.onnx.export(model_A, torch.randn(cfg.SOLVER.BATCH_SIZE, cfg.DATASET.CHANNELS, cfg.DATASET.POINTS).cuda(), "model.onnx", input_names=input_names, output_names=output_names)

    # validate
    criterion = nn.CrossEntropyLoss()
    for idx, (data, target) in enumerate(val_loader):
        data = data.float()
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output_A = model_A(data)
        loss_A = criterion(output_A, target)
        acc_A, _ = accuracy(output_A, target, topk=(1, 2))
        print(acc_A, loss_A)

        output_B = model_B(data)
        loss_B = criterion(output_B, target)
        acc_B, _ = accuracy(output_B, target, topk=(1, 2))
        print(acc_B, loss_B)

    print(model_A.inner_nodes[0].weight)
    print(model_A.leaf_nodes.weight)
