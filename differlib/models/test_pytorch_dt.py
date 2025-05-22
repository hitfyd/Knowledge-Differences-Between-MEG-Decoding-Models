import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_breast_cancer

from differlib.engine.utils import get_data_labels_from_dataset
from ncart import NCARClassifier, CART

# data = load_breast_cancer()
# X = data.data.astype(np.float32)
# y = data.target
# feature_names = data.feature_names

dataset = "CamCAN"
X, y = get_data_labels_from_dataset('../dataset/{}_train.npz'.format(dataset))
X_test, y_test = get_data_labels_from_dataset('../dataset/{}_test.npz'.format(dataset))
X, X_test = X.reshape((len(X), -1)), X_test.reshape((len(X_test), -1))


# model = NCARClassifier(epochs=100, n_trees=8, n_layers=1, n_selected=6, use_gpu=False)  # CPU
model = NCARClassifier(epochs=300, n_trees=128, n_layers=1, n_selected=96, batch_size=128)  # single GPU
# model = NCARClassifier(epochs=100, n_trees=1, n_layers=2, n_selected=2, batch_size=128, data_parallel=True, gpu_ids=[1, 2])  # multiple GPU

model.fit(X, y)

pred_y = model.predict(X)
cm = confusion_matrix(y, pred_y)
acc = accuracy_score(y, pred_y)
print(cm, acc)

pred_y_test = model.predict(X_test)
cm = confusion_matrix(y_test, pred_y_test)
acc = accuracy_score(y_test, pred_y_test)
print(cm, acc)

# importance = model.get_importance()
#
# # Create a DataFrame with feature names and importance scores
# feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
#
# # Plot feature importances using Seaborn
# plt.figure()
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df, width=0.6)
# plt.title('Feature Importances', fontsize=12)
# plt.xlabel('Importance Score', fontsize=15)
# plt.ylabel('Features', fontsize=15)
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.tight_layout()
# plt.show()
