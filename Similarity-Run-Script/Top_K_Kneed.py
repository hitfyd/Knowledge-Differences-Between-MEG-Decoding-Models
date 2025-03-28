import numpy as np
from kneed import KneeLocator
from matplotlib import pyplot as plt

# datasets
datasets = ["DecMeg2014", "CamCAN"]     # "DecMeg2014", "CamCAN"
model_names = ["MEEGNet", "Linear", "MLP", "HGRN", "LFCNN", "VARCNN", "ATCNet"]
n_x = 100

for dataset in datasets:
    for model_name in model_names:
        npz = np.load('./output/Consensus/{}/{}_{}_top_sort.npz'.format(dataset, dataset, model_name))
        top_sort, sort_contribution, sign_sort_maps = npz['abs_top_sort'], npz['abs_sort_contribution'], npz['sign_sort_maps']

        assert len(sort_contribution) % n_x == 0
        y = sort_contribution.reshape(-1, len(sort_contribution) // n_x).sum(axis=-1)
        x = range(1, n_x+1)
        # y = sort_contribution
        # x = range(1, len(sort_contribution) + 1)

        kl = KneeLocator(x, y, curve="convex", direction="decreasing")
        kl.plot_knee()
        print(dataset, model_name, kl.knee)

        plt.show()