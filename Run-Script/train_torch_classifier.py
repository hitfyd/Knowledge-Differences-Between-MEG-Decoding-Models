import os
from datetime import datetime

import torch
from torch import nn, optim

from differlib.engine.utils import get_data_labels_from_dataset, save_checkpoint, get_data_loader, setup_seed
from differlib.models import sdt
from differlib.models.DNNClassifier import mlp, linear, lfcnn, varcnn, hgrn, eegnetv1, eegnetv4


def __l1_regularization__(model, l1_penalty=3e-4):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(abs(param))  # torch.norm(param, p=1)
    return l1_penalty * regularization_loss


def train(model, train_loader, epoch, lr=3e-4, l1_penalty=0.0, l2_penalty=0.0):
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


def test(model, test_loader, validate=False):
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


# run time
run_time = datetime.now().strftime("%Y%m%d%H%M%S")

# setup the random number seed
seed = 2024
setup_seed(seed)

# trainer hyperparameters
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
# train hyperparameters
# batch_size_list = [64]
# learn_rate_list = [1e-3]
l1_penalty_list = [0.0, 0.0003]
l2_penalty_list = [0.0, 0.000001]
batch_size_list = [64, 128]     # 64, 128
learn_rate_list = [3e-4, 1e-3, 3e-3]    # 3e-4, 1e-3, 3e-3
MAX_TRAIN_EPOCHS = 200
# learn_rate_decay = 0.1
# decay_epochs = [150]

# datasets
# datasets = ["CamCAN", "DecMeg2014"]     # "DecMeg2014", "CamCAN", "ebrains", "BCIIV2a"
datasets = ["BCIIV2a"]
# models = [sdt, linear, mlp, lfcnn, varcnn, hgrn, atcnet, megnet, ctnet, eegnex, msvtnet, attentionbasenet]
# models = [lfcnn, hgrn, eegnetv1, eegnetv4, atcnet]
# models = [atcnet, mlp, linear]
# models = [sdt, lfcnn, varcnn, hgrn]
models = [eegnetv4, eegnetv1]
# physical_channels = 102
physical_channels = 22

# log config
log_path = f"./output/Train_Classifier_{run_time}/"
if not os.path.exists(log_path):
    os.makedirs(log_path)
with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
    writer.write(f"Run time: {run_time}\t Seed: {seed}\n")
    writer.write(f"batch_size_list: {batch_size_list}\tlearn_rate_list: {learn_rate_list}\t"
                 f"MAX_TRAIN_EPOCHS: {MAX_TRAIN_EPOCHS}\t"
                 # f"learn_rate_decay: {learn_rate_decay}\tdecay_epochs: {decay_epochs}\t"
                 f"datasets: {datasets}\tmodels: {models}\n")

# init dataset & models
for dataset in datasets:
    data, labels = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
    data_test, labels_test = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    _, channels, points = data.shape
    num_classes = len(set(labels_test))


    for batch_size in batch_size_list:
        train_loader = get_data_loader(data, labels, batch_size=batch_size, shuffle=True)
        test_loader = get_data_loader(data_test, labels_test)
        for learn_rate in learn_rate_list:
            for l1_penalty  in l1_penalty_list:
                for l2_penalty in l2_penalty_list:
                    for model_ in models:
                        setup_seed(seed)

                        model = model_(channels=channels, points=points, num_classes=num_classes)
                        model_name = model.__class__.__name__

                        print(f"Dataset: {dataset}\tModel: {model_name}\tLearning Rate: {learn_rate}\tBatch Size: {batch_size}\tl1_penalty: {l1_penalty}\tl2_penalty: {l2_penalty}")
                        with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
                            writer.write(f"Dataset: {dataset}\tModel: {model_name}\t"
                                         f"Learning Rate: {learn_rate}\tBatch Size: {batch_size}\tl1_penalty: {l1_penalty}\tl2_penalty: {l2_penalty}\n")

                        best_test_accuracy = 0.0
                        best_checkpoint_path = os.path.join(log_path, f"{dataset}_{model_name}_{batch_size}_{learn_rate}_"
                                                                      f"{l1_penalty}_{l2_penalty}_{run_time}_checkpoint.pt")
                        for epoch in range(MAX_TRAIN_EPOCHS):
                            # if epoch in decay_epochs:
                            #     learn_rate = learn_rate * learn_rate_decay

                            train_accuracy, train_loss = train(model, train_loader, epoch, learn_rate, l1_penalty, l2_penalty)
                            test_accuracy, test_loss = test(model, test_loader)

                            with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
                                writer.write(f"epoch: {epoch}\tlearn_rate: {learn_rate}\t"
                                             f"train_accuracy: {train_accuracy:.6f}\ttrain_loss: {train_loss:.6f}\t"
                                             f"test_accuracy: {test_accuracy:.6f}\ttest_loss: {test_loss:.6f}\n")

                            if test_accuracy > best_test_accuracy:
                                print(f'Best Test Accuracy: {best_test_accuracy:.6f} -> {test_accuracy:.6f}')
                                with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
                                    writer.write(f'Best Test Accuracy: {best_test_accuracy:.6f} -> {test_accuracy:.6f}\n')
                                best_test_accuracy = test_accuracy
                                save_checkpoint(model.state_dict(), best_checkpoint_path)
                        print(f'Dataset: {dataset}\tModel: {model_name}\t'
                              f'Learning Rate: {learn_rate}\tBatch Size: {batch_size}\tl1_penalty: {l1_penalty}\tl2_penalty: {l2_penalty}\t'
                              f'Best Test Accuracy: {best_test_accuracy:.6f}\tCheckpoint: {best_checkpoint_path}')
                        with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
                            writer.write(f'Dataset: {dataset}\tModel: {model_name}\t'
                                         f'Learning Rate: {learn_rate}\tBatch Size: {batch_size}\tl1_penalty: {l1_penalty}\tl2_penalty: {l2_penalty}\t'
                                         f'Best Test Accuracy: {best_test_accuracy:.6f}\tCheckpoint: {best_checkpoint_path}\n')
