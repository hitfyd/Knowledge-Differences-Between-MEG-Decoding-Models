import os

import numpy as np
import torch
from torch import nn, optim

from differlib.engine.utils import get_data_labels_from_dataset, save_checkpoint, get_data_loader, setup_seed
from differlib.models import model_dict
from MEG_Explanation_Comparison import top_k_consensus, top_k_disagreement


def __l1_regularization__(model, l1_penalty=3e-4):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(abs(param))  # torch.norm(param, p=1)
    return l1_penalty * regularization_loss


def train(model, train_loader, epoch, DEVICE, lr=3e-4, l1_penalty=0, l2_penalty=0):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalty)
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) + __l1_regularization__(model, l1_penalty)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_accuracy = 100. * correct / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    print('Training Dataset\tEpoch：{}\tAccuracy: [{}/{} ({:.6f}%)]\tAverage Loss: {:.6f}'.format(
        epoch, correct, len(train_loader.dataset), train_accuracy, train_loss))
    return train_accuracy, train_loss


def test(model, test_loader, DEVICE, validate=False):
    model.to(DEVICE)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            data = data.float()
            output = model(data)
            test_loss += criterion(output, target).item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    if validate:
        print('Validation Dataset\tAccuracy: {}/{} ({:.6f}%)\tAverage loss: {:.6f}'.format(
            correct, len(test_loader.dataset), test_accuracy, test_loss))
    else:
        print('Test Dataset\tAccuracy: {}/{} ({:.6f}%)\tAverage loss: {:.6f}'.format(
            correct, len(test_loader.dataset), test_accuracy, test_loss))
    # 返回测试集精度，损失
    return test_accuracy, test_loss


def train_pipeline(model_class, train_data, train_labels, test_data, test_labels, DEVICE, learn_rate, batch_size, epochs, top_k, compare_model_name=None):
    setup_seed(seed)
    model = model_class(channels=channels, points=points, num_classes=num_classes)
    model_name = model.__class__.__name__

    if compare_model_name == model_name:
        return

    train_loader = get_data_loader(train_data, train_labels, batch_size=batch_size, shuffle=True)
    test_loader = get_data_loader(test_data, test_labels)
    print(f"Dataset: {dataset}\tModel: {model_name}\tLearning Rate: {learn_rate}\tBatch Size: {batch_size}")
    # with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
    #     writer.write(f"Dataset: {dataset}\tModel: {model_name}\t"
    #                  f"Learning Rate: {learn_rate}\tBatch Size: {batch_size}\n")

    best_test_accuracy = 0.0
    best_checkpoint_path = os.path.join(log_path, f"{dataset}_{model_name}_{batch_size}_{learn_rate}_checkpoint.pt")
    for epoch in range(epochs):
        train_accuracy, train_loss = train(model, train_loader, epoch, DEVICE, learn_rate)
        test_accuracy, test_loss = test(model, test_loader, DEVICE)

        # with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
        #     writer.write(f"epoch: {epoch}\tlearn_rate: {learn_rate}\t"
        #                  f"train_accuracy: {train_accuracy:.6f}\ttrain_loss: {train_loss:.6f}\t"
        #                  f"test_accuracy: {test_accuracy:.6f}\ttest_loss: {test_loss:.6f}\n")

        if test_accuracy > best_test_accuracy:
            print(f'Best Test Accuracy: {best_test_accuracy:.6f} -> {test_accuracy:.6f}')
            with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
                writer.write(f'Best Test Accuracy: {best_test_accuracy:.6f} -> {test_accuracy:.6f}\t'
                             f'epoch: {epoch}\tlearn_rate: {learn_rate}\t'
                             f"train_accuracy: {train_accuracy:.6f}\ttrain_loss: {train_loss:.6f}\t"
                             f"test_accuracy: {test_accuracy:.6f}\ttest_loss: {test_loss:.6f}\n")
            best_test_accuracy = test_accuracy
            save_checkpoint(model.state_dict(), best_checkpoint_path)
    print(f'Dataset: {dataset}\tModel: {model_name}\tTop-k {top_k}\tCompared Model {compare_model_name}\t'
          f'Learning Rate: {learn_rate}\tBatch Size: {batch_size}\t'
          f'Best Test Accuracy: {best_test_accuracy:.6f}\tCheckpoint: {best_checkpoint_path}')
    with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
        writer.write(f'Dataset: {dataset}\tModel: {model_name}\tTop-k {top_k}\tCompared Model {compare_model_name}\t'
                     f'Learning Rate: {learn_rate}\tBatch Size: {batch_size}\t'
                     f'Best Test Accuracy: {best_test_accuracy:.6f}\tCheckpoint: {best_checkpoint_path}\n')


# setup the random number seed
seed = 2024
setup_seed(seed)

# trainer hyperparameters
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
# train hyperparameters
batch_size_list = [128]
learn_rate_list = [1e-3]
MAX_TRAIN_EPOCHS = 50

# datasets
datasets = ["DecMeg2014", "CamCAN"]     # "DecMeg2014", "CamCAN"
# top-k
top_k_list = [0.1]    # 0.05, 0.1, 0.2
model_names = ["linear", "mlp", "hgrn", "lfcnn", "varcnn", "atcnet"]    # "linear", "mlp", "hgrn", "lfcnn", "varcnn", "atcnet"
compare_model_name = "ATCNet"

consensus_all_models = False
control_train = True
consensus_train = True
disagreement_train = True

# log config
log_path = f"./output/Train_Classifier_{datasets}/"
if not os.path.exists(log_path):
    os.makedirs(log_path)
with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
    writer.write(f"Seed: {seed}\t Top-k: {top_k_list}\n")
    writer.write(f"batch_size_list: {batch_size_list}\tlearn_rate_list: {learn_rate_list}\t"
                 f"MAX_TRAIN_EPOCHS: {MAX_TRAIN_EPOCHS}\t"
                 f"datasets: {datasets}\tmodels: {model_names}\t"
                 f"compare_model_name: {compare_model_name}\n")

# init dataset & models
for dataset in datasets:
    data, labels = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
    data_test, labels_test = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    _, channels, points = data.shape
    num_classes = len(set(labels_test))

    for top_k in top_k_list:
        k = int(channels * points * top_k)

        npz_consensus_all = np.load(
            './output/Consensus_and_Disagreement_{}/{}_top_{}_union_consensus.npz'.format(dataset, dataset, top_k))
        union_consensus, union_consensus_masks = npz_consensus_all['union_consensus'], npz_consensus_all[
            'union_consensus_masks']

        for model_type in model_names:
            model_class, model_pretrain_path = model_dict[dataset][model_type]
            model = model_class(channels=channels, points=points, num_classes=num_classes)
            model_name = model.__class__.__name__
            npz = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_name))
            top_sort, sort_contribution, sign_sort_maps = npz['abs_top_sort'], npz['abs_sort_contribution'], npz['sign_sort_maps']
            npz_compare = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, compare_model_name))
            top_sort_compare = npz_compare['abs_top_sort']

            # consensus of all models 重训练
            if consensus_all_models:
                print("fixed_features:", union_consensus_masks.sum())
                fixed_data = data * union_consensus_masks
                fixed_data_test = data_test * union_consensus_masks
                for batch_size in batch_size_list:
                    for learn_rate in learn_rate_list:
                        train_pipeline(model_class, fixed_data, labels, fixed_data_test, labels_test, DEVICE, learn_rate, batch_size, MAX_TRAIN_EPOCHS, top_k)

                fixed_data = data * ~union_consensus_masks
                fixed_data_test = data_test * ~union_consensus_masks
                for batch_size in batch_size_list:
                    for learn_rate in learn_rate_list:
                        train_pipeline(model_class, fixed_data, labels, fixed_data_test, labels_test, DEVICE,
                                       learn_rate, batch_size, MAX_TRAIN_EPOCHS, top_k)

            # top-k控制组训练结果
            if control_train:
                top_masks = np.zeros_like(top_sort, dtype=np.bool_)
                top_masks[top_sort[:k]] = True
                top_masks = top_masks.reshape(channels, points)
                print("fixed_features:", top_masks.sum())
                fixed_data = data * top_masks
                fixed_data_test = data_test * top_masks
                for batch_size in batch_size_list:
                    for learn_rate in learn_rate_list:
                        train_pipeline(model_class, fixed_data, labels, fixed_data_test, labels_test, DEVICE, learn_rate, batch_size, MAX_TRAIN_EPOCHS, top_k)

            # top-k consensus with atcnet
            if consensus_train:
                consensus_list, consensus_masks = top_k_consensus(top_sort, top_sort_compare, k)
                consensus_masks = consensus_masks.reshape(channels, points)
                print("consensus_features:", len(consensus_list))
                consensus_data = data * consensus_masks
                consensus_data_test = data_test * consensus_masks # 测试集进行是否筛选对Linear、MLP影响不大，对其他模型影响明显，筛选效果更好
                for batch_size in batch_size_list:
                    for learn_rate in learn_rate_list:
                        train_pipeline(model_class, consensus_data, labels, consensus_data_test, labels_test, DEVICE, learn_rate, batch_size, MAX_TRAIN_EPOCHS, top_k, compare_model_name)

            # top-k disagreement with atcnet
            if disagreement_train:
                disagreement_list, disagreement_masks = top_k_disagreement(top_sort, top_sort_compare, k)
                disagreement_masks = disagreement_masks.reshape(channels, points)
                print("disagreement_features:", len(disagreement_list))
                disagreement_data = data * disagreement_masks
                disagreement_data_test = data_test * disagreement_masks
                for batch_size in batch_size_list:
                    for learn_rate in learn_rate_list:
                        train_pipeline(model_class, disagreement_data, labels, disagreement_data_test, labels_test, DEVICE, learn_rate, batch_size, MAX_TRAIN_EPOCHS, top_k, compare_model_name)

