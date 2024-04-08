from .am import AMethod

from imblearn.over_sampling import SMOTE

class SmoteAM(AMethod):
    def augment(self, x, y_1, y_2, *argv, **kwargs):
        oversampler = SMOTE(random_state=0)
        os_x, os_y = oversampler.fit_resample(x, y_1)
        return x, y_1, y_2
