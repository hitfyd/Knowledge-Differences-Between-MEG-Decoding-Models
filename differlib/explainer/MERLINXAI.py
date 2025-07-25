import logging

import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier, _tree

from .dise import DISExplainer
from .imd.rule import Rule
from .merlin import MERLIN


class MERLINXAI(DISExplainer):
    """
    References: L. Malandri, F. Mercorio, M. Mezzanzanica, and A. Seveso, “Model-contrastive explanations through
    symbolic reasoning,” Decis. Support Syst., vol. 176, p. 114040, Jan. 2024, doi: 10.1016/j.dss.2023.114040.
    """

    def __init__(self):
        """
        Initialize an MERLINXAI object.
        """

        super(MERLINXAI, self).__init__()

        # to be populated on calling fit() method, or set manually
        self.explainer = None
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
        self.feature_names = X_train.columns.to_list()

        Y1 = pd.Series(Y1)
        Y2 = pd.Series(Y2)

        assert Y1.shape == Y2.shape, "Y1 and Y2 must have the same"

        self.explainer = MERLIN(X_train, Y1, X_train, Y2,
                                data_type='tabular', surrogate_type='sklearn', log_level=logging.INFO,
                                hyperparameters_selection=True, save_path=f'results/',
                                save_surrogates=True, save_bdds=True)
        self.explainer.run_trace()

        # self.exp.run_explain()
        #
        # self.exp.explain.BDD2Text()
        bdds = self.explainer.trace.bdds
        id = 0
        for time_label in ['left', 'right']:
            for class_id in self.explainer.trace.classes:
                class_bdds = bdds[time_label][class_id]
                rules = class_bdds.split('|')
                for r in rules:
                    predicates = r.split('&')
                    rule = Rule(id, predicates, class_id)
                    self.diffrules.append(rule)
                    id += 1

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

        metrics[name + "_confusion_matrix"] = np.zeros((2, 2))
        metrics[name + "_accuracy"] = 0
        metrics[name + "_precision"] = 0
        metrics[name + "_recall"] = 0
        metrics[name + "_f1"] = 0

        metrics["num_rules"] = len(self.diffrules)

        preds = []
        for rule in self.diffrules:
            preds += rule.predicates
        metrics["average_num_rule_preds"] = float(len(preds)) / metrics["num_rules"]
        preds = set(preds)
        metrics["num_unique_preds"] = len(preds)
        return metrics
