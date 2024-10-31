import os
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from differlib.engine.utils import get_data_labels_from_dataset, save_checkpoint, get_data_loader, setup_seed, \
    load_checkpoint
from differlib.explainer import SeparateSurrogate, IMDExplainer, DeltaExplainer, LogitDeltaRule

# run time
run_time = datetime.now().strftime("%Y%m%d%H%M%S")

# setup the random number seed
# Data preparation
random_state = 1234
train_size = 0.7
n_times = 5
max_depth = 6
min_samples_leaf = 0.001
ccp_alpha = 0.001
setup_seed(random_state)


def disagreement_measure(out1, out2):
    n = len(out1)
    disagreement = np.sum(out1 != out2) / n
    return disagreement


# datasets
datasets = [
    "CamCAN",
    "DecMeg2014"
]
models = {
    'LR': LogisticRegression(random_state=random_state),
    # # 'KN1': KNeighborsClassifier(n_neighbors=3),
    # 'DT1': DecisionTreeClassifier(max_depth=5, random_state=random_state),
    # 'MLP1': MLPClassifier(alpha=1e-05, hidden_layer_sizes=(15,), random_state=random_state, solver='lbfgs'),
    # 'MLP2': MLPClassifier(hidden_layer_sizes=(100, 100), random_state=random_state),
    # 'DT2': DecisionTreeClassifier(max_depth=10, random_state=random_state),
    # 'GB': GradientBoostingClassifier(random_state=random_state),
    'RF1': RandomForestClassifier(random_state=random_state),
    # # 'KN2': KNeighborsClassifier(),
    # 'RF2': RandomForestClassifier(max_depth=6, random_state=random_state),
    'GNB': GaussianNB()
}
dataset_diff_models = {
    # 'CamCAN': [('MLP1', 'DT1'), ('LR', 'MLP1'), ('RF2', 'GNB'), ('DT1', 'RF2')],
    # 'DecMeg2014': [('RF2', 'GNB'), ('DT1', 'RF2'), ('MLP1', 'RF2')],
    'CamCAN': [('LR', 'RF1'),],
    'DecMeg2014': [('LR', 'RF1'), ],
}
explainers = {
    "SS": SeparateSurrogate,
    # "IMD": IMDExplainer,
}

# log config
log_path = f"./output/MEG_ML/"
if not os.path.exists(log_path):
    os.makedirs(log_path)

# init dataset & models
for dataset in datasets:
    x_train, y_train = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
    x_test_all, y_test_all = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
    _, channels, points = x_train.shape
    num_classes = len(set(y_train))
    x_train = x_train.reshape(x_train.shape[0], channels * points)
    x_test_all = x_test_all.reshape(x_test_all.shape[0], channels * points)
    x_train = pd.DataFrame(x_train)
    x_test_all = pd.DataFrame(x_test_all)

    # Training models and save checkpoints
    for model_name in models.keys():
        save_path = os.path.join(log_path, "{}_{}".format(dataset, model_name))
        if not os.path.exists(save_path):
            model = models[model_name]
            model.fit(x_train, y_train)
            save_checkpoint(model, save_path)
        else:
            model = load_checkpoint(save_path)
        models[model_name] = model

        t_acc = accuracy_score(y_true=y_test_all, y_pred=model.predict(x_test_all))
        print(f"dataset: {dataset} model: {model_name} test accuracy: {(t_acc * 100):.2f}%")
        with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
            writer.write(f"dataset: {dataset} model: {model_name} test accuracy: {(t_acc * 100):.2f}%" + os.linesep)

    # Computing differencing
    for model1_name, model2_name in dataset_diff_models[dataset]:
        model1, model2 = models[model1_name], models[model2_name]

        for explainer_type in explainers.keys():
            pd_test_metrics, pd_train_metrics = None, None
            pd_accuracy_list, pd_disagreement_list = None, None
            # skf = StratifiedShuffleSplit(n_splits=n_times, test_size=0.25)
            # skf_id = 0
            # for train_index, test_index in skf.split(x_test_all, y_test_all):
            #     x_train, x_test, y_train, y_test = x_test_all.iloc[train_index], x_test_all.iloc[test_index], y_test_all[train_index], y_test_all[test_index]

            for skf_id in range(n_times):
                x_train, x_test, y_train, y_test = train_test_split(x_test_all, y_test_all, train_size=train_size, random_state=random_state+skf_id)

                # Calculate diff-samples
                y1 = model1.predict(x_train)
                y2 = model2.predict(x_train)
                output1 = model1.predict_proba(x_train)
                output2 = model2.predict_proba(x_train)
                ydiff = (y1 != y2).astype(int)
                print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff)):.2f}")

                t_y1 = model1.predict(x_test)
                t_y2 = model2.predict(x_test)
                t_output1 = model1.predict_proba(x_test)
                t_output2 = model2.predict_proba(x_test)
                ydifftest = (t_y1 != t_y2).astype(int)
                print(
                    f"diffs in X_test = {ydifftest.sum()} / {len(ydifftest)} = {(ydifftest.sum() / len(ydifftest)):.2f}")

                explainer = explainers[explainer_type]()
                jstobj, t1, t2 = explainer.fit_detail(x_train, y1, y2, max_depth, min_samples_leaf=min_samples_leaf,
                                                      ccp_alpha=ccp_alpha)
                train_metrics = explainer.metrics(x_train, y1, y2, name="train")
                test_metrics = explainer.metrics(x_test, t_y1, t_y2)

                y_surrogate1 = jstobj.predict(x_test.to_numpy(), t1)
                y_surrogate2 = jstobj.predict(x_test.to_numpy(), t2)

                disagreement_list = {
                    'origin1_accuracy': sklearn.metrics.accuracy_score(y_test, t_y1),
                    'origin2_accuracy': sklearn.metrics.accuracy_score(y_test, t_y2),
                    'surrogate1_accuracy': sklearn.metrics.accuracy_score(y_test, y_surrogate1),
                    'surrogate2_accuracy': sklearn.metrics.accuracy_score(y_test, y_surrogate2),
                    'origin1_origin2': disagreement_measure(t_y1, t_y2),
                    'origin1_surrogate2': disagreement_measure(t_y1, y_surrogate2),
                    'surrogate1_origin2': disagreement_measure(y_surrogate1, t_y2),
                    'surrogate1_surrogate2': disagreement_measure(y_surrogate1, y_surrogate2),
                    'origin1_surrogate1': disagreement_measure(t_y1, y_surrogate1),
                    'origin2_surrogate2': disagreement_measure(t_y2, y_surrogate2),
                }

                # 打印单次实验结果
                print(dataset, explainer_type, model1_name, model2_name, skf_id)
                print(train_metrics)
                print(test_metrics)
                print(disagreement_list)

                # 记录单次实验的训练和测试结果
                train_metrics['train-confusion_matrix'] = np.array2string(train_metrics['train-confusion_matrix'])
                if pd_train_metrics is None:
                    pd_train_metrics = pd.DataFrame(columns=train_metrics.keys())
                pd_train_metrics.loc[len(pd_train_metrics)] = train_metrics.values()

                test_metrics['test-confusion_matrix'] = np.array2string(test_metrics['test-confusion_matrix'])
                if pd_test_metrics is None:
                    pd_test_metrics = pd.DataFrame(columns=test_metrics.keys())
                pd_test_metrics.loc[len(pd_test_metrics)] = test_metrics.values()

                if pd_disagreement_list is None:
                    pd_disagreement_list = pd.DataFrame(columns=disagreement_list.keys())
                pd_disagreement_list.loc[len(pd_disagreement_list)] = disagreement_list.values()

            print(dataset, explainer_type, model1_name, model2_name, "mean_std")
            # 计算disagreement相似度
            pd_disagreement_list_mean, pd_disagreement_list_std = pd_disagreement_list.mean(), pd_disagreement_list.std()
            disagreement_mean_std = pd.Series(index=pd_disagreement_list_mean.index, dtype=str)
            for v in range(len(pd_disagreement_list_mean.values)):
                disagreement_mean_std.iloc[
                    v] = f"{pd_disagreement_list_mean.iloc[v]:.2f} ± {pd_disagreement_list_std.iloc[v]:.2f}"
            print(disagreement_mean_std.to_string())

            # 计算测试集上各个指标的均值和标准差
            assert len(pd_test_metrics.columns.tolist()) == 10
            partial_pd_metrics = pd_test_metrics.iloc[:, 3:]
            partial_pd_metrics_mean, partial_pd_metrics_std = partial_pd_metrics.mean(), partial_pd_metrics.std()
            record_mean_std = pd.Series(index=partial_pd_metrics_mean.index, dtype=str)
            for v in range(len(partial_pd_metrics_mean.values)):
                record_mean_std.iloc[
                    v] = f"{partial_pd_metrics_mean.iloc[v]:.2f} ± {partial_pd_metrics_std.iloc[v]:.2f}"
            print(record_mean_std.to_string())
            with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
                writer.write(os.linesep + "-" * 25 + os.linesep)
                writer.write(f"{dataset} {model1_name} {model2_name} {explainer_type}" + os.linesep)
                writer.write(pd_train_metrics.to_string() + os.linesep)
                writer.write(pd_test_metrics.to_string() + os.linesep)
                writer.write(record_mean_std.to_string() + os.linesep)
                writer.write(pd_disagreement_list.to_string() + os.linesep)
                writer.write(disagreement_mean_std.to_string() + os.linesep)
                writer.write(os.linesep + "-" * 25 + os.linesep)

            # # 根据模型A、B，记录不同解释器配置下的测试集实验结果用于对比
            # record_file = os.path.join(log_path, f"{dataset}_{model1_name}_{model2_name}_record.csv")
            # record_mean_std['model1'] = model1_name
            # record_mean_std['model2'] = model2_name
            # record_mean_std['explainer'] = explainer_type
            # if os.path.exists(record_file):
            #     all_record_mean_std = pd.read_csv(record_file)
            #     assert all_record_mean_std.columns.tolist() == record_mean_std.index.tolist()
            # else:
            #     all_record_mean_std = pd.DataFrame(columns=record_mean_std.index)
            # all_record_mean_std.loc[len(all_record_mean_std)] = record_mean_std.values
            # all_record_mean_std.to_csv(record_file, index=False)
