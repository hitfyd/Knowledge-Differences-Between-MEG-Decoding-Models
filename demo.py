import os
import logging
from zipfile import ZipFile
import pandas as pd
from differlib.explainer.merlin import MERLIN

with ZipFile('tests/test_data/20newsgroups.zip', 'r') as archive:
    df_left = pd.read_csv(archive.open('df_left.csv'), delimiter=',')
    df_right = pd.read_csv(archive.open('df_right.csv'), delimiter=',')

df_left = df_left[~df_left['corpus'].isnull()]
df_right = df_right[~df_right['corpus'].isnull()]

# Load data
X_left, Y_left, predicted_labels_left = df_left[
    'corpus'], df_left['category'], df_left['predicted_labels']
X_right, Y_right, predicted_labels_right = df_right[
    'corpus'], df_right['category'], df_right['predicted_labels']

exp = MERLIN(X_left, predicted_labels_left,
             X_right, predicted_labels_right,
             data_type='text', surrogate_type='sklearn', log_level=logging.INFO,
             hyperparameters_selection=True, save_path=f'results/',
             save_surrogates=True, save_bdds=True)

percent_dataset = 1
print(
    f'Running Trace with percent of dataset to use: {percent_dataset}', flush=True)
exp.run_trace(percent_dataset)

exp.run_explain()

exp.explain.BDD2Text()
