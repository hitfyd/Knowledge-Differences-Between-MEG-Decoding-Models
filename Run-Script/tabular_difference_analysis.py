import os

import arff
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from differlib.engine.utils import save_checkpoint, load_checkpoint
from differlib.explainer import DeltaExplainer, IMDExplainer, LogitDeltaRule, SeparateSurrogate


def load_bc_dataset():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    return data["data"], data["target"]


def load_magic_dataset():
    df = pd.read_csv("../dataset/tabular/magic04.data", header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values
    y = [0 if y[i] == 'g' else 1 for i in range(len(y))]
    return X, np.array(y)


def load_waveform_dataset():
    # data = arff.load("../dataset/tabular/waveform-5000.arff")
    # df = pd.DataFrame(data)
    df = pd.read_csv("../dataset/tabular/waveform-+noise.data", header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values
    # y = [0 if y[i] == 'N' else 1 for i in range(len(y))]
    return X, np.array(y)


def load_heloc_dataset():
    data = arff.load("../dataset/tabular/dataset_HELOC")
    df = pd.DataFrame(data)
    X = df.iloc[:, 1:]
    X[[10, 11]] = X[[10, 11]].astype(int)
    y = df.iloc[:, 0].values
    y = np.squeeze(y)
    y = [0 if y[i] == 'Bad' else 1 for i in range(len(y))]
    return X, np.array(y)


def load_bank_marketing_dataset():
    data = arff.load("../dataset/tabular/dataset_bank-marketing")
    df = pd.DataFrame(data)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values
    y = [0 if y[i] == '1' else 1 for i in range(len(y))]
    return X, np.array(y)


def load_eye_movements_dataset():
    data = arff.load("../dataset/tabular/dataset_eye_movements")
    df = pd.DataFrame(data)
    X = df.iloc[:, :-1]
    X[[20, 21, 22]] = X[[20, 21, 22]].astype(int)
    y = df.iloc[:, -1].values
    return X, y


# def load_whitewine_dataset():
#     # data = arff.load("../dataset/tabular/whitewine.arff")
#     # df = pd.DataFrame(data)
#     df = pd.read_csv("../dataset/tabular/winequality-white.csv", delimiter=';')
#     X = df.iloc[:, :-1]
#     y = df.iloc[:, -1].values
#     y = y.astype(int) - 3
#     return X, np.array(y)


def load_redwine_dataset():
    # data = arff.load("../dataset/tabular/wine-quality-red.arff")
    # df = pd.DataFrame(data, dtype=float)
    df = pd.read_csv("../dataset/tabular/winequality-red.csv", delimiter=';')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values
    y = y.astype(int) - 3
    return X, np.array(y)


def load_parkinsons_dataset():
    df = pd.read_csv("../dataset/tabular/parkinsons.data")
    X = df.iloc[:, 1:]
    X.pop('status')
    y = df.iloc[:, 17].values
    return X, np.array(y)


# Data preparation
random_state = 1234
train_size = 0.7
n_times = 5
max_depth = 6
min_samples_leaf = 1
ccp_alpha = 0.001
datasets = {
    'bc': load_bc_dataset(),
    'parkinsons': load_parkinsons_dataset(),
    'eye_movements': load_eye_movements_dataset(),
    'waveform': load_waveform_dataset(),
    'heloc': load_heloc_dataset(),
    'bank_marketing': load_bank_marketing_dataset(),
    'magic': load_magic_dataset(),
    'redwine': load_redwine_dataset(),
}
models = {
    'LR': LogisticRegression(random_state=random_state),
    'KN1': KNeighborsClassifier(n_neighbors=3),
    'DT1': DecisionTreeClassifier(max_depth=5, random_state=random_state),
    'MLP1': MLPClassifier(alpha=1e-05, hidden_layer_sizes=(15,), random_state=random_state, solver='lbfgs'),
    'MLP2': MLPClassifier(hidden_layer_sizes=(100, 100), random_state=random_state),
    'DT2': DecisionTreeClassifier(max_depth=10, random_state=random_state),
    'GB': GradientBoostingClassifier(random_state=random_state),
    'RF1': RandomForestClassifier(random_state=random_state),
    'KN2': KNeighborsClassifier(),
    'RF2': RandomForestClassifier(max_depth=6, random_state=random_state),
    'GNB': GaussianNB()
}
dataset_diff_models = {
    'bank_marketing': [('MLP2', 'GB'), ('LR', 'KN2')],
    'bc': [('DT1', 'GNB'), ('KN2', 'RF2')],
    'eye_movements': [('RF1', 'MLP1'), ('KN1', 'DT1')],     # min应改为KN1-KN2，但是由于两者模型架构一致，改为次min对KN1-DT1
    'heloc': [('KN1', 'GB'), ('GB', 'RF1')],
    'magic': [('RF1', 'GNB'), ('MLP2', 'DT1')],
    'waveform': [('LR', 'DT2'), ('MLP1', 'RF2')],
    'whitewine': [('RF1', 'GNB'), ('LR', 'KN2')],
    'redwine': [('RF1', 'KN2'), ('LR', 'MLP1')],
    'parkinsons': [('RF1', 'GNB'), ('MLP1', 'LR')],
}
explainers = {
    "SS": SeparateSurrogate,
    "IMD": IMDExplainer,
    "Delta": DeltaExplainer,
    "Logit": LogitDeltaRule,
    # "MERLIN": MERLINXAI,
}

log_path = './output/tabular/'
if not os.path.exists(log_path):
    os.makedirs(log_path)

for dataset in datasets.keys():
    data, target = datasets[dataset]
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=train_size, random_state=random_state)

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

        t_acc = accuracy_score(y_true=y_test, y_pred=model.predict(x_test))
        print(f"dataset: {dataset} model: {model_name} test accuracy: {(t_acc * 100):.2f}%")
        with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
            writer.write(f"dataset: {dataset} model: {model_name} test accuracy: {(t_acc * 100)}%" + os.linesep)

    # Computing differencing
    for model1_name, model2_name in dataset_diff_models[dataset]:
        model1, model2 = models[model1_name], models[model2_name]

        for explainer_type in explainers.keys():
            pd_test_metrics, pd_train_metrics = None, None
            for skf_id in range(n_times):
                x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=train_size, random_state=random_state+skf_id)

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
                print(f"diffs in X_test = {ydifftest.sum()} / {len(ydifftest)} = {(ydifftest.sum() / len(ydifftest)):.2f}")

                explainer = explainers[explainer_type]()
                if explainer_type in ["Logit"]:
                    explainer.fit(x_train, output1, output2, max_depth, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha)
                    train_metrics = explainer.metrics(x_train, output1, output2, name="train")
                    test_metrics = explainer.metrics(x_test, t_output1, t_output2)
                else:
                    explainer.fit(x_train, y1, y2, max_depth, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha)
                    train_metrics = explainer.metrics(x_train, y1, y2, name="train")
                    test_metrics = explainer.metrics(x_test, t_y1, t_y2)

                # 打印单次实验结果
                print(dataset, explainer_type, model1_name, model2_name, skf_id)
                print(train_metrics)
                print(test_metrics)

                # 记录单次实验的训练和测试结果
                train_metrics['train-confusion_matrix'] = np.array2string(train_metrics['train-confusion_matrix'])
                if pd_train_metrics is None:
                    pd_train_metrics = pd.DataFrame(columns=train_metrics.keys())
                pd_train_metrics.loc[len(pd_train_metrics)] = train_metrics.values()

                test_metrics['test-confusion_matrix'] = np.array2string(test_metrics['test-confusion_matrix'])
                if pd_test_metrics is None:
                    pd_test_metrics = pd.DataFrame(columns=test_metrics.keys())
                pd_test_metrics.loc[len(pd_test_metrics)] = test_metrics.values()

            # 计算测试集上各个指标的均值和标准差
            assert len(pd_test_metrics.columns.tolist()) == 10
            partial_pd_metrics = pd_test_metrics.iloc[:, 3:]
            partial_pd_metrics_mean, partial_pd_metrics_std = partial_pd_metrics.mean(), partial_pd_metrics.std()
            record_mean_std = pd.Series(index=partial_pd_metrics_mean.index, dtype=str)
            for v in range(len(partial_pd_metrics_mean.values)):
                record_mean_std.iloc[v] = f"{partial_pd_metrics_mean.iloc[v]:.2f} ± {partial_pd_metrics_std.iloc[v]:.2f}"
            print(record_mean_std.to_string())
            with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
                writer.write(os.linesep + "-" * 25 + os.linesep)
                writer.write(f"{dataset} {model1_name} {model2_name} {explainer_type}" + os.linesep)
                writer.write(pd_train_metrics.to_string() + os.linesep)
                writer.write(pd_test_metrics.to_string() + os.linesep)
                writer.write(partial_pd_metrics_mean.to_string() + os.linesep)
                writer.write(partial_pd_metrics_std.to_string() + os.linesep)
                writer.write(record_mean_std.to_string() + os.linesep)
                writer.write(os.linesep + "-" * 25 + os.linesep)

            # 根据模型A、B，记录不同解释器配置下的测试集实验结果用于对比
            record_file = os.path.join(log_path, f"{dataset}_{model1_name}_{model2_name}_record.csv")
            record_mean_std['model1'] = model1_name
            record_mean_std['model2'] = model2_name
            record_mean_std['explainer'] = explainer_type
            if os.path.exists(record_file):
                all_record_mean_std = pd.read_csv(record_file, encoding="utf_8_sig")
                assert all_record_mean_std.columns.tolist() == record_mean_std.index.tolist()
            else:
                all_record_mean_std = pd.DataFrame(columns=record_mean_std.index)
            all_record_mean_std.loc[len(all_record_mean_std)] = record_mean_std.values
            all_record_mean_std.to_csv(record_file, index=False, encoding="utf_8_sig")
