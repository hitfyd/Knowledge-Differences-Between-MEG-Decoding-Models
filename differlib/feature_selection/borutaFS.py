import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

from .fsm import FSMethod


class BorutaFS(FSMethod):
    def __init__(self):
        super(BorutaFS, self).__init__()
        perc = 80
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        self.method = BorutaPy(rf, n_estimators='auto', perc=perc, alpha=0.05, two_step=True, max_iter=30, verbose=2)
        self.contributions = None

    def fit(self, x: np.ndarray, pred_target_A, pred_target_B, *args, **kwargs):
        assert pred_target_A.shape == pred_target_B.shape
        if len(pred_target_A.shape) == 2:
            pred_target_A = pred_target_A.argmax(axis=1)
            pred_target_B = pred_target_B.argmax(axis=1)
        delta_target = pred_target_A ^ pred_target_B
        self.method.fit(x, delta_target)

    def computing_contribution(self, *argv, **kwargs):
        pass

    def transform(self, x: np.ndarray, *args, **kwargs):
        return self.method.transform(x)
