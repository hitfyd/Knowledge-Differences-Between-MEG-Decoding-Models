import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from aix360.algorithms.imd.utils import load_bc_dataset

# Data preparation
random_state=1234
datadf, target = load_bc_dataset()
x_train, x_test, y_train, y_test = train_test_split(datadf, target, train_size=0.7,
                                                    random_state=random_state)
x_train.shape, x_test.shape

# Training models
## model1
model1 = DecisionTreeClassifier(max_depth=5)
model1.fit(x_train, y_train)
print(f"model: {model1}")
tacc = accuracy_score(y_true=y_test, y_pred=model1.predict(x_test))
print(f"model1 test accuracy: {(tacc * 100):.2f}%")

## model2
model2 = GaussianNB()
model2.fit(x_train, y_train)

print(f"model: {model2}")
tacc = accuracy_score(y_true=y_test, y_pred=model2.predict(x_test))
print(f"model2 test accuracy: {(tacc * 100):.2f}%")

# Calculate diff-samples %
feature_names = x_train.columns.to_list()
x1 = x2 = x_train.to_numpy()
y1 = model1.predict(x1)
y2 = model2.predict(x2)
ydiff = (y1 != y2).astype(int)
print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum()/len(ydiff)):.2f}")

ydifftest = (model1.predict(x_test) != model2.predict(x_test)).astype(int)
print(f"diffs in X_test = {ydifftest.sum()} / {len(ydifftest)} = {(ydifftest.sum()/len(ydifftest)):.2f}")

# Interpretable model differencing
from aix360.algorithms.imd.imd import IMDExplainer

max_depth=6

imd = IMDExplainer()
imd.fit(x_train, y1, y2, max_depth=max_depth)

# See diff-rules
diffrules = imd.explain()
print(diffrules)

# Using the diff-rules
rule_idx = 4

rule = diffrules[rule_idx]
filtered_data = x_train[rule.apply(x_train)]
print(filtered_data)

# Computation of metrics
# on train set
imd.metrics(x_train, y1, y2, name="train")

# on test set
imd.metrics(x_test, model1.predict(x_test), model2.predict(x_test), name="test")

# Generate the jst visualization
from aix360.algorithms.imd.utils import visualize_jst
visualize_jst(imd.jst, path="idm_example_joint.jpg")

# Separate surrogate approach
sepsur = IMDExplainer()
sepsur.fit(x_train, y1, y2, max_depth=max_depth, split_criterion=2, alpha=1.0)

# on train set
sepsur.metrics(x_train, y1, y2, name="train")

# on test set
sepsur.metrics(x_test, model1.predict(x_test), model2.predict(x_test), name="test")

# Visualizing separate surrogates
visualize_jst(sepsur.jst, path="idm_example_separate.jpg")
