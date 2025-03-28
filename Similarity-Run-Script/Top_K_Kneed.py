import numpy as np
from kneed import KneeLocator
from matplotlib import pyplot as plt

# datasets
datasets = ["DecMeg2014", "CamCAN"]     # "DecMeg2014", "CamCAN"
model_names = ["Linear", "MLP", "HGRN", "LFCNN", "VARCNN", "ATCNet"]    # "MEEGNet",
n_x = 100
s = 10

for dataset in datasets:
    # 模型kneed结果聚合
    # plt.style.use("ggplot")
    plt.figure(figsize=(6, 4))
    title = f"Knee Point (Dataset: {dataset})"
    xlabel = "Percents of Top Contribution Features(%)"
    ylabel = "Contribution"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(0, 101, 10))
    colors = ["gold", "springgreen","moccasin",  "navy", "orchid", "salmon"]

    knees = []
    for model_name in model_names:
        npz = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_name))
        top_sort, sort_contribution, sign_sort_maps = npz['abs_top_sort'], npz['abs_sort_contribution'], npz['sign_sort_maps']

        assert len(sort_contribution) % n_x == 0
        y = sort_contribution.reshape(-1, len(sort_contribution) // n_x).sum(axis=-1)
        x = range(1, n_x+1)
        # y = sort_contribution
        # x = range(1, len(sort_contribution) + 1)

        y = (y - np.min(y)) / (np.max(y) - np.min(y))   # 贡献经过了MaxMinScalar
        kl = KneeLocator(x, y, curve="convex", direction="decreasing", S=s)
        knee = kl.knee
        print(dataset, model_name, knee)
        # figsize = (5, 4)
        # title = f"Knee Point = {knee}(Dataset: {dataset}, Model: {model_name})"
        # xlabel = "Percents of Top Contribution Features(%)"
        # ylabel = "Contribution"
        # kl.plot_knee(figsize, title, xlabel, ylabel)
        # plt.savefig(f"./output/Consensus/{dataset}_{model_name}_{knee}.svg")
        # plt.show()

        knees.append(knee)
        plt.plot(kl.x, kl.y, label=f"Contribution (Model: {model_name})")

    final_knees = []
    final_model_names = []
    for i in range(len(knees)):
        if knees[i] not in final_knees:
            final_knees.append(knees[i])
            final_model_names.append(model_names[i])
        else:
            j = final_knees.index(knees[i])
            final_model_names[j] = f"{final_model_names[j]}, {model_names[i]}"

    for k, c, m in zip(final_knees, colors, final_model_names):
        plt.vlines(k, 0, 1, linestyles="--", colors=c, label=f"k = {k} (Model: {m})")
    plt.legend()
    plt.savefig(f"./output/Consensus/{dataset}_kneed.svg")
    plt.show()
