from catboost import CatBoostClassifier, Pool
import pandas as pd
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import neptune.new as neptune
import optuna
import logging

K_FOLDS = 3
IMPORTANT_FEATURES = 256

dataset = pd.read_csv('../hackaton_input.csv', index_col=0)
features_ranked_by_importance = list(pd.read_csv('sorted_by_importance.csv', index_col=0)["0"])
contFeatures = list(set(dataset.columns) - {'Activity'})
values = dataset[contFeatures].values
col_mean = np.nanmean(values, axis=0)
inds = np.where(np.isnan(values))
values[inds] = np.take(col_mean, inds[1])
dataset[contFeatures] = values

# make test set for k-fold
subjects = list(dataset.subject.unique())
labels = list(dataset.Activity.unique())
final_test_subjects = [10,11,12]
subjects = list(set(subjects) - set(final_test_subjects))
final_test_dataset = dataset[dataset['subject'].isin(final_test_subjects)]
dataset = dataset[dataset['subject'].isin(subjects)]

subjects_permutation = np.random.permutation(subjects)
test_subjects_number = int(len(subjects) / K_FOLDS)
important_features = features_ranked_by_importance[:IMPORTANT_FEATURES]

params = {
  "learning_rate": 0.20281366269439707,
  "l2_leaf_reg": 1,
  "depth": 7,
  "bagging_temperature": 0.38532733636500577,
  "random_strength": 0.6907090863574054,
  "loss_function": 'MultiClass',
  "task_type": "GPU",
  "eval_metric":'Accuracy',
  "logging_level": "Silent"
}
models = [CatBoostClassifier(**params) for _ in range(K_FOLDS)]
  
for fold_index in range(K_FOLDS):
  test_subjects = np.roll(subjects_permutation, test_subjects_number * fold_index)[:test_subjects_number]
  train_subjects = list(set(subjects) - set(test_subjects))

  train_dataset = dataset[dataset['subject'].isin(train_subjects)]
  test_dataset = dataset[dataset['subject'].isin(test_subjects)]

  train_X = train_dataset[important_features].values
  test_X = test_dataset[important_features].values
  
  train_y = train_dataset['Activity'].values
  test_y = test_dataset['Activity'].values

  model = models[fold_index]

  model.fit(train_X, train_y, eval_set=Pool(test_X, test_y))

  preds_class = model.predict(Pool(test_X, test_y))

  print(accuracy_score(test_y, preds_class))

def test(models):
  labels.sort()
  test_X = final_test_dataset[important_features].values
  test_y = final_test_dataset['Activity'].values
  predictions = []
  for model in models:
    preds_proba = model.predict_proba(Pool(test_X, test_y))
    predictions.append(preds_proba)
  predictions = np.array(predictions)
  preds = np.argmax(np.mean(predictions, axis=0), axis=1)
  final_test_preds = np.array([labels[pred] for pred in preds])
  print(accuracy_score(test_y, final_test_preds))

for idx, model in enumerate(models):
  model.save_model('model_{}_weights'.format(idx))
test(models)

