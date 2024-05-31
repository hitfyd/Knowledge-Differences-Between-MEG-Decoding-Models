import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from differlib.engine.utils import save_checkpoint, load_checkpoint
from differlib.explainer import DeltaExplainer, IMDExplainer, LogitDeltaRule, explainer_dict, SeparateSurrogate, \
    MERLINXAI
from differlib.explainer.imd.utils import load_bc_dataset, load_waveform_dataset

# Data preparation
random_state = 1234
train_size = 0.7
n_times = 5
max_depth = 4
min_samples_leaf = 1
datasets = {'bc': load_bc_dataset(),
            'waveform': load_waveform_dataset(),
            }
models = {'LR': LogisticRegression(random_state=random_state),
          'DT1': DecisionTreeClassifier(max_depth=5),
          # 'GB': GradientBoostingClassifier(),
          'NB': GaussianNB()
          }
explainers = {"SS": SeparateSurrogate,
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
        t_acc = accuracy_score(y_true=y_test, y_pred=model.predict(x_test))
        print(f"model: {model_name} test accuracy: {(t_acc * 100):.2f}%")
        models[model_name] = model

    n_models = len(models.keys())
    model_types_list = list(models.keys())
    models_list = list(models.values())
    for i in range(n_models-1):
        for j in range(i+1, n_models):
            model1, model2 = models_list[i], models_list[j]
            model1_name, model2_name = model_types_list[i], model_types_list[j]

            # Calculate diff-samples %
            # feature_names = x_train.columns.to_list()
            # x1 = x2 = x_train.to_numpy()
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

            for explainer_type in explainers.keys():
                pd_test_metrics, pd_train_metrics = None, None
                for skf_id in range(n_times):
                    # x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=train_size)
                    explainer = explainers[explainer_type]()
                    if explainer_type in ["Logit"]:
                        explainer.fit(x_train, output1, output2, max_depth, min_samples_leaf=min_samples_leaf,)
                        train_metrics = explainer.metrics(x_train, output1, output2, name="train")
                        test_metrics = explainer.metrics(x_test, t_output1, t_output2)
                    else:
                        explainer.fit(x_train, y1, y2, max_depth, min_samples_leaf=min_samples_leaf)
                        train_metrics = explainer.metrics(x_train, y1, y2, name="train")
                        test_metrics = explainer.metrics(x_test, t_y1, t_y2)

                    # 打印单次实验结果
                    print(dataset, explainer_type, model1, model2, skf_id)
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
                    writer.write(pd_train_metrics.to_string() + os.linesep)
                    writer.write(pd_test_metrics.to_string() + os.linesep)
                    writer.write(record_mean_std.to_string() + os.linesep)
                    writer.write(os.linesep + "-" * 25 + os.linesep)

                # 根据模型A、B，记录不同解释器配置下的测试集实验结果用于对比
                record_file = os.path.join(log_path, f"{dataset}_{model1_name}_{model2_name}_record.csv")
                record_mean_std['model1'] = model1_name
                record_mean_std['model2'] = model2_name
                record_mean_std['explainer'] = explainer_type
                if os.path.exists(record_file):
                    all_record_mean_std = pd.read_csv(record_file)
                    assert all_record_mean_std.columns.tolist() == record_mean_std.index.tolist()
                else:
                    all_record_mean_std = pd.DataFrame(columns=record_mean_std.index)
                all_record_mean_std.loc[len(all_record_mean_std)] = record_mean_std.values
                all_record_mean_std.to_csv(record_file, index=False)
