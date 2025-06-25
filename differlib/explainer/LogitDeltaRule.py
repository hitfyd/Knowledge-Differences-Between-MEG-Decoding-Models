import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt, gridspec
from numpy.ma.core import argmax
from sklearn.tree import _tree, DecisionTreeRegressor, plot_tree

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
        self.hyperparameters = {}
        self.diffrules = []
        self.feature_names = []
        self.num_classes = 0

    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        pass

    def fit(self, X_train: pd.DataFrame, Y1, Y2, max_depth, min_samples_leaf=1, ccp_alpha=0.001, verbose=False, feature_weights=None, **kwargs):
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
        self.feature_names = X_train.columns.to_list()

        # X_train = X_train.to_numpy()

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
        delta_proxy_output = np.hstack((delta_output, Y1, Y2))
        sample_weight = abs(delta_output).sum(axis=1)
        self.num_classes = Y1.shape[1]

        self.delta_tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha)
        self.delta_tree.fit(X_train, delta_proxy_output, sample_weight=sample_weight)

        # if delta_output.shape[1] == 2:
        #     self.delta_tree.fit(X_train, delta_output, sample_weight=abs(delta_output[:, 0]))
        # else:
        #     label_delta_output = abs(delta_output).sum(axis=0)
        #     self.delta_tree.fit(X_train, delta_output, sample_weight=abs(delta_output[:, argmax(label_delta_output)]))  # abs(delta_output).sum(axis=1)
        self.diffrules = dtree_to_rule(self.delta_tree, feature_names=self.feature_names)
        fig = plt.figure(figsize=(5, 5))
        gridlayout = gridspec.GridSpec(ncols=25, nrows=6, figure=fig, top=None, bottom=None, wspace=None, hspace=0)
        axs1 = fig.add_subplot(gridlayout[:, :24])
        plot_tree(self.delta_tree, feature_names=self.feature_names, label='none', impurity=False, ax=axs1)
        format_list = ["eps", "pdf", "svg"]
        plt.rcParams['savefig.dpi'] = 300  # 图片保存像素
        for save_format in format_list:
            fig.savefig('1.{}'.format(save_format), format=save_format,bbox_inches='tight', transparent=False)
        # print(self.diffrules)

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
        y_logit_delta = y_test1 - y_test2

        metrics = {}
        diff_samples = pred_target_1 != pred_target_2
        total_number_diff_samples = np.sum(diff_samples)
        metrics["diffs"] = total_number_diff_samples
        metrics["samples"] = len(x_test)

        delta_target = (pred_target_1 != pred_target_2).astype(int)
        logit_delta_proxy = self.delta_tree.predict(x_test)
        logit_delta = logit_delta_proxy[:, :self.num_classes]
        logit_proxy_y1 = logit_delta_proxy[:, self.num_classes:self.num_classes *2]
        logit_proxy_y2 = logit_delta_proxy[:, self.num_classes*2:]
        y_test2_ = y_test1 - logit_delta
        pred_target = (pred_target_1 != y_test2_.argmax(axis=1)).astype(int)
        # y_test1_ = y_test2 + logit_delta
        # pred_target = (y_test1_.argmax(axis=1) != pred_target_2).astype(int)
        # pred_target = (logit_proxy_y1.argmax(axis=1) != logit_proxy_y2.argmax(axis=1)).astype(int)
        # if len(set(pred_target_1)) > 2:
        #     delta_target = (pred_target_1 ^ pred_target_2).astype(int)
        #     pred_target = (pred_target_1 ^ y_test2_.argmax(axis=1)).astype(int)

        # calcuate loss
        squared_loss = sklearn.metrics.mean_squared_error(logit_delta, y_logit_delta)
        print("squared_loss", squared_loss)
        print("Proxy Model1", "Acc:", sklearn.metrics.accuracy_score(pred_target_1, logit_proxy_y1.argmax(axis=1)), "MSE:", sklearn.metrics.mean_squared_error(y_test1, logit_proxy_y1))
        print("Proxy Model2", "Acc:", sklearn.metrics.accuracy_score(pred_target_2, logit_proxy_y2.argmax(axis=1)), "MSE:", sklearn.metrics.mean_squared_error(y_test2, logit_proxy_y2))

        metrics[name + "-confusion_matrix"] = sklearn.metrics.confusion_matrix(delta_target, pred_target)
        metrics[name + "-accuracy"] = sklearn.metrics.accuracy_score(delta_target, pred_target)
        metrics[name + "-precision"] = sklearn.metrics.precision_score(delta_target, pred_target)#, average='weighted')
        metrics[name + "-recall"] = sklearn.metrics.recall_score(delta_target, pred_target)#, average='weighted')
        metrics[name + "-f1"] = sklearn.metrics.f1_score(delta_target, pred_target)#, average='weighted')

        metrics["num-rules"] = len(self.diffrules)

        preds = []
        for rule in self.diffrules:
            preds += rule.predicates
        metrics["average-num-rule-preds"] = 0 if metrics["num-rules"] == 0 else float(len(preds)) / metrics["num-rules"]
        preds = set(preds)
        metrics["num-unique-preds"] = len(preds)
        return metrics
