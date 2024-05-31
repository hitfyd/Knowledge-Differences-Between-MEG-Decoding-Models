import os

import numpy as np
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
max_depth = 5
min_samples_leaf = 1
datasets = {'bc': load_bc_dataset(),
            'waveform': load_waveform_dataset(),}
models = {'LR': LogisticRegression(random_state=random_state),
          'DT1': DecisionTreeClassifier(max_depth=5),
          # 'GB': GradientBoostingClassifier(),
          'NB': GaussianNB()}
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
    models_list = list(models.values())
    for i in range(n_models-1):
        for j in range(i+1, n_models):
            model1, model2 = models_list[i], models_list[j]

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
                explainer = explainers[explainer_type]()
                if explainer_type in ["Logit"]:
                    explainer.fit(x_train, output1, output2, max_depth, min_samples_leaf=min_samples_leaf,)
                    train_metrics = explainer.metrics(x_train, output1, output2, name="train")
                    test_metrics = explainer.metrics(x_test, t_output1, t_output2)
                else:
                    explainer.fit(x_train, y1, y2, max_depth, min_samples_leaf=min_samples_leaf)
                    train_metrics = explainer.metrics(x_train, y1, y2, name="train")
                    test_metrics = explainer.metrics(x_test, t_y1, t_y2)

                print(dataset, explainer_type, model1, model2)
                print(train_metrics)
                print(test_metrics)
