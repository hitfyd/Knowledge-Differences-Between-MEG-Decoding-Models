from sklearn.neural_network import MLPClassifier

from differlib.engine.utils import get_data_labels_from_dataset

dataset = "DecMeg2014"
# init dataset & models
data, labels = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
data_test, labels_test = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
n_samples, channels, points = data.shape
data = data.reshape((-1, channels * points))
data_test = data_test.reshape((-1, channels * points))

clf = MLPClassifier(random_state=1, max_iter=300).fit(data, labels)
print(clf.predict_proba(data_test[:1]))
print(clf.predict(data_test[:5, :]))
print(clf.score(data_test, labels_test))
