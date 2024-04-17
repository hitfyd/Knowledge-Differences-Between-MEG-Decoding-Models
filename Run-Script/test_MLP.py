import torch
from torch import nn, optim

from differlib.engine.utils import get_data_labels_from_dataset, get_data_loader_from_dataset, save_checkpoint, \
    get_data_loader
from differlib.models.DNNClassifier import mlp
from differlib.models.transformer.atcnet import atcnet
from differlib.models.transformer.eegconformer import conformer


def train(model, train_loader, epoch, lr=3e-4, l1_penalty=0, l2_penalty=0):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalty)
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
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


dataset = "CamCAN"
# dataset = "DecMeg2014"
# init dataset & models
data, labels = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
data_test, labels_test = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
_, channels, points = data.shape
classes = len(set(labels_test))

train_loader = get_data_loader(data, labels, batch_size=512, shuffle=True)
test_loader = get_data_loader(data_test, labels_test)

# model = mlp(channels, points, classes)
# learn_rate = 0.00003
# MAX_TRAIN_EPOCHS = 50
model = conformer(channels, points, classes)
learn_rate = 0.01
MAX_TRAIN_EPOCHS = 50
# model = atcnet(channels, points, classes)
# learn_rate = 0.001
# MAX_TRAIN_EPOCHS = 50

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

best_test_accuracy = 0.0
best_checkpoint_path = f"../checkpoint/Models_Train/{dataset}_{model.__class__.__name__}_best_checkpoint.pt"
for epoch in range(MAX_TRAIN_EPOCHS):
    train_accuracy, train_loss = train(model, train_loader, epoch, learn_rate)
    test_accuracy, test_loss = test(model, test_loader, validate=True)
    if test_accuracy > best_test_accuracy:
        print('Best Test Accuracy: {:.6f} -> {:.6f}'.format(best_test_accuracy, test_accuracy))
        best_test_accuracy = test_accuracy
        save_checkpoint(model.state_dict(), best_checkpoint_path)
print('Best Test Accuracy: {:.6f}'.format(best_test_accuracy))
