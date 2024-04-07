import numpy as np
import pandas as pd
import sklearn

from .fsm import FSMethod


class LinearRegression(FSMethod):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.method = sklearn.linear_model.LinearRegression()
        self.contributions = None

    def fit(self, x: np.ndarray, logit1, logit2, *args, **kwargs):
        assert len(logit1.shape) == 2 and logit1.shape == logit2.shape, "logit1 and logit2 must have the same"
        delta_logit = logit1 - logit2

        self.method.fit(x, delta_logit[:, 0])
        self.contributions = self.method.coef_

    def computing_contribution(self, **kwargs):
        return self.contributions

    def transform(self, x: np.ndarray, *args, rate=0.1, **kwargs):
        assert len(x.shape) == 2
        kth = int(len(self.contributions) * rate)
        ind = np.argpartition(self.contributions, kth=-kth)[-kth:]
        threshold = np.min(self.contributions[ind])
        print(kth, threshold)
        return x[:, ind]

    def metrics(self, x_test: pd.DataFrame, logit1, logit2, name="test"):
        assert len(logit1.shape) == len(logit2.shape) == 2 and logit1.shape == logit2.shape, \
            "logit1 and logit2 must have the same"
        pred_target_1 = logit1.argmax(axis=1)
        pred_target_2 = logit2.argmax(axis=1)

        metrics = {}
        diff_samples = pred_target_1 != pred_target_2
        total_number_diff_samples = np.sum(diff_samples)
        metrics["diffs"] = total_number_diff_samples
        metrics["samples"] = len(x_test)

        delta_target = (pred_target_1 != pred_target_2).astype(int)
        logit_delta = self.LinearRegression.predict(x_test)
        y_test2_ = logit1 - np.array([logit_delta, -logit_delta]).swapaxes(1, 0)
        pred_target_2_ = y_test2_.argmax(axis=1)
        pred_target = pred_target_1 ^ pred_target_2_
        metrics[name + "-confusion_matrix"] = sklearn.metrics.confusion_matrix(delta_target, pred_target)
        metrics[name + "-accuracy"] = sklearn.metrics.accuracy_score(delta_target, pred_target)
        metrics[name + "-precision"] = sklearn.metrics.precision_score(delta_target, pred_target)
        metrics[name + "-recall"] = sklearn.metrics.recall_score(delta_target, pred_target)
        metrics[name + "-f1"] = sklearn.metrics.f1_score(delta_target, pred_target)
        return metrics


class SGDRegressor(LinearRegression):
    def __init__(self):
        super(SGDRegressor, self).__init__()
        self.method = sklearn.linear_model.SGDRegressor()
        self.contributions = None
