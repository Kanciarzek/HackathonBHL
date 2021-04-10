import pandas as pd

kaggle_csv = pd.read_csv('../kaggle_input.csv')

hackaton_csv = pd.read_csv('../hackaton_input.csv')


for key in hackaton_csv.keys():
    if key in kaggle_csv.keys():
        continue
    print(key)