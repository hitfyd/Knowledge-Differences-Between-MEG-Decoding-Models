from .am import AMethod

from imblearn.over_sampling import SMOTE


class SmoteAM(AMethod):
    def augment(self, data, labels, *argv, **kwargs):
        n_samples, channels, points = data.shape
        oversampler = SMOTE(random_state=0)
        rs_data, rs_labels = oversampler.fit_resample(data.reshape((-1, channels * points)), labels)
        return rs_data.reshape((-1, channels, points)), rs_labels
