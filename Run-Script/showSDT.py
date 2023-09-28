import os
import argparse

import numpy as np
import sympy
import torch
# import graphviz
# import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from graphviz import Digraph
from matplotlib.collections import LineCollection
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.distributed.pipeline.sync.copy import Copy

# import torchsummary
# from torchviz import make_dot

from differlib.engine.cfg import CFG as cfg
from differlib.engine.utils import log_msg, setup_seed, get_data_loader_from_dataset, load_checkpoint, accuracy, \
    get_data_labels_from_dataset
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
    val_data, val_labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(cfg.DATASET.TYPE))

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


    def retain_model(model, lower_len_first=100, upper_len_first=100):
        inner_weights = model.inner_nodes[0].weight[:, 1:]
        for i in range(len(inner_weights)):
            inner_weight = inner_weights[i]
            lower_len = lower_len_first
            upper_len = upper_len_first
            # lower_len = lower_len_first * (2 ** int(sympy.log(i+1, 2)))
            # upper_len = upper_len_first * (2 ** int(sympy.log(i+1, 2)))
            # lower_len = int(lower_len_first / (10 ** int(sympy.log(i + 1, 2))))
            # upper_len = int(upper_len_first / (10 ** int(sympy.log(i + 1, 2))))
            inner_weight_sort, _ = inner_weight.sort()
            lower_bound = inner_weight_sort[lower_len - 1]
            upper_bound = inner_weight_sort[-upper_len]
            mask = inner_weight >= upper_bound
            mask += inner_weight <= lower_bound
            result = torch.mul(mask, inner_weight)
            print(i, lower_bound, upper_bound)
            with torch.no_grad():
                model.inner_nodes[0].weight[i, 1:] = result


    retain_model(model_A)
    retain_model(model_B)

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
    targets, pred_A, pred_B = [], [], []
    criterion = nn.CrossEntropyLoss()
    for idx, (data, target) in enumerate(val_loader):
        targets.extend(target.numpy())
        data = data.float()
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output_A = model_A(data)
        _, pred_target = output_A.topk(1, 1, True, True)
        pred_target = pred_target.squeeze().cpu().detach().numpy()
        pred_A.extend(pred_target)
        acc_A = accuracy_score(y_true=target.cpu().detach().numpy(), y_pred=pred_target)
        print(f"{idx}: model_A test accuracy: {(acc_A * 100):.2f}%")
        # loss_A = criterion(output_A, target)
        # acc_A, _ = accuracy(output_A, target, topk=(1, 2))
        # print(acc_A, loss_A)

        output_B = model_B(data)
        _, pred_target = output_B.topk(1, 1, True, True)
        pred_target = pred_target.squeeze().cpu().detach().numpy()
        pred_B.extend(pred_target)
        acc_B = accuracy_score(y_true=target.cpu().detach().numpy(), y_pred=pred_target)
        print(f"{idx}: model_B test accuracy: {(acc_B * 100):.2f}%")
        # loss_B = criterion(output_B, target)
        # acc_B, _ = accuracy(output_B, target, topk=(1, 2))
        # print(acc_B, loss_B)

    acc_A = accuracy_score(y_true=targets, y_pred=pred_A)
    print(f"sum_model_A test accuracy: {(acc_A * 100):.2f}%")
    cm_A = confusion_matrix(y_true=targets, y_pred=pred_A)
    print('cm_A is:\n', cm_A)

    acc_B = accuracy_score(y_true=targets, y_pred=pred_B)
    print(f"sum_model_B test accuracy: {(acc_B * 100):.2f}%")
    cm_B = confusion_matrix(y_true=targets, y_pred=pred_B)
    print('cm_B is:\n', cm_B)


    def plot_SDT(model, name='model'):
        inner_weights = model.inner_nodes[0].weight.cpu().detach().numpy()
        leaf_weights = model.leaf_nodes.weight.cpu().detach().numpy()
        print(inner_weights.shape)
        print(leaf_weights.shape)

        # 绘制软决策树
        dot = Digraph()

        # inner_weights = (255 * (inner_weights - np.nanmin(inner_weights)) /
        #                  (np.nanmax(inner_weights) - np.nanmin(inner_weights)))  # 归一化
        for inner_i in range(len(inner_weights)):
            inner_weight = inner_weights[inner_i][1:]
            inner_weight = inner_weight.reshape(cfg.DATASET.CHANNELS, cfg.DATASET.POINTS)
            # 使用Image保存图片
            # inner_weight = inner_weight.astype(np.uint8)
            # img = Image.fromarray(inner_weight)
            # img.save("temp/{}_{}.eps".format(name, inner_i))
            # 使用matplotlib保存图片

            # # 绘制时间曲线图
            # fig = plt.figure(figsize=(2, 4))
            # axs = fig.add_subplot()
            # cmap = 'plasma'
            # channels = cfg.DATASET.CHANNELS
            # points = cfg.DATASET.POINTS
            # thespan = np.percentile(inner_weight, 98)
            # xx = np.arange(1, points + 1)
            #
            # for channel in range(channels):
            #     y = inner_weight[channel, :] + thespan * (channels - 1 - channel)
            #     dydx = inner_weight[channel, :]
            #
            #     img_points = np.array([xx, y]).T.reshape(-1, 1, 2)
            #     segments = np.concatenate([img_points[:-1], img_points[1:]], axis=1)
            #     lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(-1, 1), linewidths=(1,))
            #     lc.set_array(dydx)
            #     axs.add_collection(lc)

            plt.imshow(inner_weight, cmap="plasma")  # gray, plasma
            plt.axis('off')
            plt.gcf().set_size_inches(cfg.DATASET.POINTS / 100.0, cfg.DATASET.CHANNELS / 100.0)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.savefig("temp/{}_{}.svg".format(name, inner_i))

            # 绘制内部节点
            dot.node(name="inner_{}".format(inner_i), label="", image="temp/{}_{}.svg".format(name, inner_i),
                     shape="square", color="white", style="filled")
        leaf_weights = ((leaf_weights - np.nanmin(leaf_weights)) /
                        (np.nanmax(leaf_weights) - np.nanmin(leaf_weights)))  # 归一化
        for leaf_j in range(leaf_weights.shape[1]):
            leaf_weight = leaf_weights[:, leaf_j]
            leaf_max_label = np.argmax(leaf_weight)
            print(leaf_weight, leaf_max_label)

            # 绘制叶子节点
            dot.node(name="leaf_{}".format(leaf_j), label="{}".format(leaf_max_label), color='blue')
            # dot.node(name="leaf_{}".format(leaf_j), label="{}\n{}".format(np.around(leaf_weight, 2), leaf_max_label),
            #          color='blue')
        # 绘制连接线
        for inner_i in range(len(inner_weights) // 2):
            dot.edge("inner_{}".format(inner_i), "inner_{}".format(inner_i * 2 + 1))
            dot.edge("inner_{}".format(inner_i), "inner_{}".format(inner_i * 2 + 2))
        first_leaf_father = len(inner_weights) // 2
        for inner_i in range(len(inner_weights) // 2, len(inner_weights)):
            dot.edge("inner_{}".format(inner_i), "leaf_{}".format((inner_i - first_leaf_father) * 2))
            dot.edge("inner_{}".format(inner_i), "leaf_{}".format((inner_i - first_leaf_father) * 2 + 1))

        dot.render(filename=name, format="pdf")


    # plot_SDT(model_A, cfg.MODELS.A)
    # plot_SDT(model_B, cfg.MODELS.B)

    # 举例绘制一个样本
