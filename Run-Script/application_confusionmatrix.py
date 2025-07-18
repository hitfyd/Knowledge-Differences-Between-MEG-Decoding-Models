import sklearn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from differlib.engine.utils import dataset_info_dict, get_data_labels_from_dataset, save_figure
from differlib.models import load_pretrained_model, output_predict_targets


dataset_list = ['CamCAN', 'DecMeg2014']
model_types = ["rf", "varcnn", "hgrn", "atcnet"]
model_names= {
    'rf': 'Random Forest',
    'varcnn': 'VARCNN',
    'hgrn': 'HGRN',
    'atcnet': 'ATCNet',
}
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')     # 'cuda:1'
print(f"Using device: {device}")

# 设置全局字体大小
plt.rcParams['font.size'] = 16
# plt.rcParams['font.weight'] = 'bold'
fig, axes = plt.subplots(
    len(dataset_list), len(model_types), figsize=(4 * len(model_types), 4 * len(dataset_list)), gridspec_kw={'hspace': 0.32, 'wspace': 0.15}  # 增加行间距，减少列间距
)
fig.suptitle(f'Accuracy and Confusion Matrix of Models', fontsize=20, fontweight='bold', y=0.98)
fig.subplots_adjust(top=0.88, bottom=0.1)

# 创建子图编号标签
subplot_labels = []
label_counter = 0
for i in range(len(dataset_list)):
    row_labels = []
    for j in range(len(model_types)):
        row_labels.append(f'({chr(97 + label_counter)})')  # 97是'a'的ASCII码
        label_counter += 1
    subplot_labels.append(row_labels)

for idx, dataset in enumerate(dataset_list):
    data, labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    label_names = ['Audio', 'Visual']
    if dataset == 'DecMeg2014':
        label_names = ['Scramble', 'Face']
    channels, points, num_classes = dataset_info_dict[dataset].values()
    for jdx, model_type in enumerate(model_types):
        model = load_pretrained_model(model_type, dataset, channels, points, num_classes, device)
        pred_output, pred_targets = output_predict_targets(model_type, model, data, num_classes=num_classes, device=device)
        cm = sklearn.metrics.confusion_matrix(labels, pred_targets)
        acc = sklearn.metrics.accuracy_score(labels, pred_targets)
        print(dataset, model_type, cm, acc)
        ax = axes[idx, jdx]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(include_values=True, cmap='Oranges',  colorbar=False, ax=ax, text_kw={"fontsize": 18, "fontweight": "bold"})
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('')
        disp.ax_.set_title(f'{model_names[model_type]} (Acc={acc*100:.2f}%)', fontsize=16, pad=12)
        if jdx != 0:
            disp.ax_.set_yticks([])

        # 设置子图标题和标签
        if jdx == 0:
            ax.set_ylabel(dataset, fontsize=18, fontweight='bold', labelpad=14, x=0.05)

        # 在子图下角添加编号
        ax.text(0.5, -0.1, subplot_labels[idx][jdx],
                transform=ax.transAxes,
                fontsize=16,
                fontweight='bold',
                verticalalignment='top')

figure_name = f"All_ConfusionMatrix"
save_figure(fig, './images/', figure_name)
