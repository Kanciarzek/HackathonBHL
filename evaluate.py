from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.metrics import accuracy_score


df = pd.read_csv('../datasets/final_train.csv', index_col=0)
train_df = df[df['subject'] <= 21].copy()
test_df = df[df['subject'] > 21].copy()
important_features_full = list(pd.read_csv("./sorted_by_importance.csv", index_col=0)["0"])
important_features = important_features_full[:249]

train_df_r = train_df[['Activity'] + important_features]
test_df_r = test_df[['Activity'] + important_features]

test_X = test_df_r.drop('Activity', axis='columns').values
test_y = test_df_r['Activity'].values

model = CatBoostClassifier(
    learning_rate=0.20281366269439707,
    l2_leaf_reg=1,
    depth=7,
    random_strength=0.6907090863574054,
    bagging_temperature=0.38532733636500577,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    random_seed=2501
)

model.load_model('model_weights')
# make the prediction using the resulting model
preds_class = model.predict(Pool(test_X, test_y))

print(accuracy_score(test_y, preds_class))

