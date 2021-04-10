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

if sys.argv[-1] == "neptune":
    run = neptune.init(
                project='Naukowe-Kolo-Robotyki-I-Sztucznej-Inteligencji/best-hacking-league',
                name='first-pytorch-ever')

df = pd.read_csv('../dataset/final_train.csv', index_col=0)
train_df = df[df['subject'] <= 21].copy()
test_df = df[df['subject'] > 21].copy()
important_features_full = list(pd.read_csv("./sorted_by_importance.csv", index_col=0)["0"])
important_features = important_features_full[:249]

train_df_r = train_df[['Activity'] + important_features]
test_df_r = test_df[['Activity'] + important_features]
train_X = train_df_r.drop('Activity', axis='columns').values
train_y = train_df_r['Activity'].values
test_X = test_df_r.drop('Activity', axis='columns').values
test_y = test_df_r['Activity'].values


train_Z = train_X.copy()
col_mean = np.nanmean(train_X, axis=0)
inds = np.where(np.isnan(train_X))
train_Z[inds] = np.take(col_mean, inds[1])
X_resampled, y_resampled = SMOTE(random_state=2501).fit_resample(train_Z, train_y)

def objective():
    model = CatBoostClassifier(
        learning_rate=0.20281366269439707,
        l2_leaf_reg=1,
        depth=7,
        random_strength=0.6907090863574054,
        bagging_temperature=0.38532733636500577,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=2501,
        #task_type='GPU'
    )

    # train the model
    model.fit(X_resampled, y_resampled, eval_set=Pool(test_X, test_y))

    # make the prediction using the resulting model
    preds_class = model.predict(Pool(test_X, test_y))

    model.save_model('model_weights')

    return accuracy_score(test_y, preds_class)

objective()

#optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=1000)

# model.save_model('model_weights')

#     run['parameters'] = model.get_params()
#     run['model_checkpoints/my_model'].upload('model_weights')
