import numpy as np
import pandas as pd
import json

dict_to_json = dict()
df = pd.read_csv("IMDB Dataset.csv")

for _, row in df.iterrows():
    dict_to_json[row['review']] = row['sentiment']

with open('IMDB_Dataset.json', 'w') as f:
    json.dump(dict_to_json, f)
