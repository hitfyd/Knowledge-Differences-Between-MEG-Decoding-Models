import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier, _tree

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
            class_label = class_labels[value.argmax()]
            # print("node {} depth {} parent {} class_label {}".format(node, depth, parent, class_label))
            # print("value {} n_node_samples {} impurity {}".format(value, n_node_samples, impurity))
            # print("predicates {}".format(predicates))
            rule = Rule(node, predicates[parent], class_label)
            rule_list.append(rule)

    recurse(0, 1, 0)
    return rule_list


class DeltaExplainer(DISExplainer):
    """
    DeltaXplainer, a model-agnostic method for generating rule-based explanations describing the differences between
    two binary classifiers.

    References: A. Rida, M.-J. Lesot, X. Renard, and C. Marsala, “Dynamic Interpretability
    for Model Comparison via Decision Rules.” arXiv, Sep. 29, 2023. Accessed: Oct. 07, 2023. [Online]. Available:
    https://arxiv.org/abs/2309.17095

    """

    def __init__(self):
        """
        Initialize an DeltaXplainer object.
        """

        super(DeltaExplainer, self).__init__()

        # to be populated on calling fit() method, or set manually
        self.delta_tree = None
        self.diffrules = []
        self.feature_names = []

    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        pass

    def fit(self, X_train: pd.DataFrame, Y1, Y2, max_depth, min_samples_leaf=1, verbose=True, feature_names = None, **kwargs):
        """
        Fit joint surrogate tree to input data, and outputs from two models.
        Args:
            X_train: input dataframe
            Y1: model1 outputs
            Y2: model2 outputs
            max_depth: maximum depth of the joint surrogate tree to be built
            min_samples_leaf: minimum number of samples required to be at a leaf node
            verbose:
            **kwargs:
        Returns:
            self
        """
        if feature_names is None:
            feature_names = X_train.columns.to_list()
        self.feature_names = feature_names

        X_train = X_train.to_numpy()

        if not isinstance(Y1, np.ndarray):
            Y1 = Y1.to_numpy()
        if not isinstance(Y2, np.ndarray):
            Y2 = Y2.to_numpy()

        assert Y1.shape == Y2.shape, "Y1 and Y2 must have the same"

        ydiff = (Y1 != Y2).astype(int)
        if verbose:
            print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff) * 100):.2f}%")

        delta_target = (Y1 != Y2).astype(int)

        self.delta_tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, ccp_alpha=0.001)
        self.delta_tree.fit(X_train, delta_target)
        self.diffrules = dtree_to_rule(self.delta_tree, feature_names=self.feature_names)
        # print(export_text(self.delta_tree, feature_names=self.feature_names, show_weights=True))
        # plot_tree(self.delta_tree)

    def predict(self, X, *argv, **kwargs):
        """Predict diff-labels.
        """
        pass

    def explain(self, *argv, **kwargs):
        """Return diff-rules.
        """
        return self.diffrules

    def metrics(self, x_test: pd.DataFrame, y_test1, y_test2, name="test"):

        metrics = {}
        diff_samples = y_test1 != y_test2
        total_number_diff_samples = np.sum(diff_samples)
        metrics["diffs"] = total_number_diff_samples
        metrics["samples"] = len(x_test)

        delta_target = (y_test1 != y_test2).astype(int)
        pred_target = self.delta_tree.predict(x_test)
        metrics[name + "-confusion_matrix"] = sklearn.metrics.confusion_matrix(delta_target, pred_target)
        metrics[name + "-accuracy"] = sklearn.metrics.accuracy_score(delta_target, pred_target)
        metrics[name + "-precision"] = sklearn.metrics.precision_score(delta_target, pred_target)
        metrics[name + "-recall"] = sklearn.metrics.recall_score(delta_target, pred_target)
        metrics[name + "-f1"] = sklearn.metrics.f1_score(delta_target, pred_target)

        metrics["num-rules"] = len(self.diffrules)

        preds = []
        for rule in self.diffrules:
            preds += rule.predicates
        metrics["average-num-rule-preds"] = float(len(preds)) / metrics["num-rules"]
        preds = set(preds)
        metrics["num-unique-preds"] = len(preds)
        return metrics

