from sklearn import tree

from differlib.dise import DISExplainer

import numpy as np
import pandas as pd


def dtree_to_rule(tree, feature_names, df_name='data', badrate_need=0.5):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != tree.tree_.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    pathto = dict()
    global k
    k = 0
    rule_list = []

    def recurse(node, depth, parent, badrate_need):
        global k
        if tree_.feature[node] != tree.tree_.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            s = "({}['{}'] <= {})".format(df_name, name, threshold)
            if node == 0:
                pathto[node] = s
            else:
                pathto[node] = pathto[parent] + ' & ' + s

            recurse(tree_.children_left[node], depth + 1, node, badrate_need)
            s = "({}['{}'] > {})".format(df_name, name, threshold)
            if node == 0:
                pathto[node] = s
            else:
                pathto[node] = pathto[parent] + ' & ' + s

            recurse(tree_.children_right[node], depth + 1, node, badrate_need)
        else:
            k = k + 1
            dct = {}
            if tree_.value[node][0][1] / tree_.n_node_samples[node] >= badrate_need:
                dct['rule_name'] = pathto[parent]
                dct['bad_rate'] = tree_.value[node][0][1] / tree_.n_node_samples[node]
                dct['bad_num'] = tree_.value[node][0][1]
                dct['hit_num'] = tree_.n_node_samples[node]
                # sum(tree_.value[0][0]) # total
                rule_list.append(dct)

    recurse(0, 1, 0, badrate_need)
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

    def fit(self, X_train: pd.DataFrame, Y1, Y2, max_depth, min_samples_leaf=1, verbose=True, **kwargs):
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
        feature_names = X_train.columns.to_list()
        self.feature_names = feature_names

        X_train = X_train.to_numpy()

        if not isinstance(Y1, np.ndarray):
            Y1 = Y1.to_numpy()
        if not isinstance(Y2, np.ndarray):
            Y2 = Y2.to_numpy()

        ydiff = (Y1 != Y2).astype(int)
        if verbose:
            print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff) * 100):.2f}%")

        delta_target = Y1 ^ Y2

        self.delta_tree = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.delta_tree.fit(X_train, delta_target)
        self.diffrules = tree.export_text(self.delta_tree, feature_names=self.feature_names, show_weights=True)

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
        # y_test1 = self.model1.predict(x_test)
        # y_test2 = self.model2.predict(x_test)
        diff_samples = y_test1 != y_test2
        total_number_diff_samples = np.sum(diff_samples)
        metrics["diffs"] = total_number_diff_samples
        metrics["samples"] = len(x_test)

        # inregiondiff = self.inregion(self.diffregions, x_test[diff_samples].to_numpy())
        # diff_samples_inside_diff_region = np.sum(inregiondiff)
        # inregion = self.inregion(self.diffregions, x_test.to_numpy())
        # samples_in_region = np.sum(inregion)
        #
        # metrics[name + "-precision"] = round(diff_samples_inside_diff_region / samples_in_region, 6)
        # metrics[name + "-recall"] = round(diff_samples_inside_diff_region / total_number_diff_samples, 6)
        metrics["num-rules"] = len(self.diffrules)

        preds = []
        for rule in self.diffrules:
            preds += rule.predicates
        preds = set(preds)
        metrics["num-unique-preds"] = len(preds)
        return metrics

