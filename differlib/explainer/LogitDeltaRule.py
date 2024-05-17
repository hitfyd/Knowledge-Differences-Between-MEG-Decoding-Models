import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import _tree, DecisionTreeRegressor
from catboost import CatBoostRegressor

from .dise import DISExplainer
from .imd.rule import Rule


def dtree_to_rule(tree, feature_names, class_labels=[0, 1]):
    tree_ = tree.tree_

    predicates = dict()
    assert len(class_labels) == 2
    rule_list = []

    def recurse(node, depth, parent):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            key = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            pred = (key, '<=', threshold)
            if node == 0:
                predicates[node] = [pred]
            else:
                predicates[node] = []
                predicates[node].extend(predicates[parent])
                predicates[node].append(pred)

            recurse(tree_.children_left[node], depth + 1, node)
            pred = (key, '>', threshold)
            if node == 0:
                predicates[node] = [pred]
            else:
                predicates[node] = []
                predicates[node].extend(predicates[parent])
                predicates[node].append(pred)

            recurse(tree_.children_right[node], depth + 1, node)
        else:
            value = tree_.value[node].squeeze()
            n_node_samples = tree_.n_node_samples[node]
            impurity = tree_.impurity[node]
            # print("node {} depth {} parent {}".format(node, depth, parent))
            # print("value {} n_node_samples {} impurity {}".format(value, n_node_samples, impurity))
            # print("predicates {}".format(predicates))
            rule = Rule(node, predicates[parent], value)
            rule_list.append(rule)

    recurse(0, 1, 0)
    return rule_list


class LogitDeltaRule(DISExplainer):
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

        super(LogitDeltaRule, self).__init__()

        # to be populated on calling fit() method, or set manually
        self.delta_tree = None
        self.diffrules = []
        self.feature_names = []

    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        pass

    def fit(self, X_train: pd.DataFrame, Y1, Y2, max_depth, min_samples_leaf=1, verbose=True, feature_weights=None, **kwargs):
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

        assert Y1.shape == Y2.shape, "Y1 and Y2 must have the same"
        pred_target_1 = Y1.argmax(axis=1)
        pred_target_2 = Y2.argmax(axis=1)

        ydiff = (pred_target_1 != pred_target_2).astype(int)
        if verbose:
            print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff) * 100):.2f}%")

        delta_target = (pred_target_1 != pred_target_2).astype(int)
        delta_output = Y1 - Y2

        # self.delta_tree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, ccp_alpha=0.001)
        # self.delta_tree.fit(X_train, delta_output, sample_weight=abs(delta_output[:, 0]))
        # self.diffrules = dtree_to_rule(self.delta_tree, feature_names=self.feature_names)
        # print(export_text(self.delta_tree, feature_names=self.feature_names, show_weights=True))
        # plot_tree(self.delta_tree)

        # self.delta_tree = GradientBoostingRegressor(n_estimators=5, min_samples_leaf=min_samples_leaf, max_depth=None, ccp_alpha=0.001)
        # self.delta_tree.fit(X_train, delta_output[:, 0], sample_weight=abs(delta_output[:, 0]))
        # self.diffrules = dtree_to_rule(self.delta_tree.estimators_[0][0], feature_names=self.feature_names)

        self.delta_tree = CatBoostRegressor(iterations=100, depth=None, learning_rate=0.1, loss_function='RMSE', task_type="GPU", devices='0', feature_weights=feature_weights)
        self.delta_tree.fit(X_train, delta_output[:, 0], sample_weight=abs(delta_output[:, 0]))
        self.diffrules = []

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
        # y_test2_ = y_test1 - logit_delta
        y_test2_ = np.zeros_like(y_test1)
        y_test2_[:, 0] = y_test1[:, 0] - logit_delta
        y_test2_[:, 1] = y_test1[:, 1] + logit_delta
        pred_target_2_ = y_test2_.argmax(axis=1)
        pred_target = pred_target_1 ^ pred_target_2_
        metrics[name + "-confusion_matrix"] = sklearn.metrics.confusion_matrix(delta_target, pred_target)
        metrics[name + "-accuracy"] = sklearn.metrics.accuracy_score(delta_target, pred_target)
        metrics[name + "-precision"] = sklearn.metrics.precision_score(delta_target, pred_target)
        metrics[name + "-recall"] = sklearn.metrics.recall_score(delta_target, pred_target)
        metrics[name + "-f1"] = sklearn.metrics.f1_score(delta_target, pred_target)

        metrics["num-rules"] = len(self.diffrules)

        preds = []
        for rule in self.diffrules:
            preds += rule.predicates
        metrics["average-num-rule-preds"] = 0 if metrics["num-rules"] == 0 else float(len(preds)) / metrics["num-rules"]
        preds = set(preds)
        metrics["num-unique-preds"] = len(preds)
        return metrics

