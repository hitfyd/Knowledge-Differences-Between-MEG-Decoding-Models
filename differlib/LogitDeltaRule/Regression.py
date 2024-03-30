import sklearn
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge

from differlib.dise import DISExplainer

import numpy as np
import pandas as pd


class Regression(DISExplainer):
    """
    DeltaXplainer, a model-agnostic method for generating rule-based explanations describing the differences between
    two binary classifiers.

    References: A. Rida, M.-J. Lesot, X. Renard, and C. Marsala, “Dynamic Interpretability
    for Model Comparison via Decision Rules.” arXiv, Sep. 29, 2023. Accessed: Oct. 07, 2023. [Online]. Available:
    https://arxiv.org/abs/2309.17095

    """

    def __init__(self):
        """
        Initialize an LogitDeltaRule object.
        """

        super(Regression, self).__init__()

        # to be populated on calling fit() method, or set manually
        self.delta_tree = None
        self.diffrules = []
        self.feature_names = []

    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        pass

    def fit(self, X_train: pd.DataFrame, Y1, Y2, max_depth, min_samples_leaf=1, verbose=True, **kwargs):
        """
        Fit joint surrogate tree to input data, and outputs from two models.
        Args:
            X_train: input dataframe
            Y1: model1 outputs(logits)
            Y2: model2 outputs(logits)
            max_depth: maximum depth of the joint surrogate tree to be built
            min_samples_leaf: minimum number of samples required to be at a leaf node
            verbose:
            **kwargs:
        Returns:
            self
        """
        feature_names = X_train.columns.to_list()
        self.feature_names = feature_names

        X_train = X_train.to_numpy()

        if not isinstance(Y1, np.ndarray):
            Y1 = Y1.to_numpy()
        if not isinstance(Y2, np.ndarray):
            Y2 = Y2.to_numpy()

        assert len(Y1.shape) == len(Y2.shape) == 2 and Y1.shape[0] == Y2.shape[0], "Y1 and Y2 must have the same"
        pred_target_1 = Y1.argmax(axis=1)
        pred_target_2 = Y2.argmax(axis=1)

        ydiff = (pred_target_1 != pred_target_2).astype(int)
        if verbose:
            print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff) * 100):.2f}%")

        delta_target = (pred_target_1 != pred_target_2).astype(int)
        delta_output = Y1 - Y2

        # self.delta_tree = LinearRegression()
        # self.delta_tree = Ridge()
        self.delta_tree = SGDRegressor()
        self.delta_tree.fit(X_train, delta_output[:, 0])
        self.diffrules = []
        print(self.delta_tree.coef_)

    def predict(self, X, *argv, **kwargs):
        """Predict diff-labels.
        """
        pass

    def explain(self, *argv, **kwargs):
        """Return diff-rules.
        """
        return self.diffrules

    def metrics(self, x_test: pd.DataFrame, y_test1, y_test2, name="test"):
        assert len(y_test1.shape) == len(y_test2.shape) == 2 and y_test1.shape[0] == y_test2.shape[0], \
            "y_test1 and y_test2 must have the same"
        pred_target_1 = y_test1.argmax(axis=1)
        pred_target_2 = y_test2.argmax(axis=1)

        metrics = {}
        diff_samples = pred_target_1 != pred_target_2
        total_number_diff_samples = np.sum(diff_samples)
        metrics["diffs"] = total_number_diff_samples
        metrics["samples"] = len(x_test)

        delta_target = (pred_target_1 != pred_target_2).astype(int)
        logit_delta = self.delta_tree.predict(x_test)
        y_test2_ = y_test1 - np.array([logit_delta, -logit_delta]).swapaxes(1, 0)
        pred_target_2_ = y_test2_.argmax(axis=1)
        pred_target = pred_target_1 ^ pred_target_2_
        metrics[name + "-confusion_matrix"] = sklearn.metrics.confusion_matrix(delta_target, pred_target)
        metrics[name + "-accuracy"] = sklearn.metrics.accuracy_score(delta_target, pred_target)
        metrics[name + "-precision"] = sklearn.metrics.precision_score(delta_target, pred_target)
        metrics[name + "-recall"] = sklearn.metrics.recall_score(delta_target, pred_target)
        metrics[name + "-f1"] = sklearn.metrics.f1_score(delta_target, pred_target)

        # metrics["num-rules"] = len(self.diffrules)
        #
        # preds = []
        # for rule in self.diffrules:
        #     preds += rule.predicates
        # metrics["average-num-rule-preds"] = float(len(preds)) / metrics["num-rules"]
        # preds = set(preds)
        # metrics["num-unique-preds"] = len(preds)
        metrics["num-rules"] = 0
        metrics["average-num-rule-preds"] = 0
        metrics["num-unique-preds"] = 0
        return metrics

