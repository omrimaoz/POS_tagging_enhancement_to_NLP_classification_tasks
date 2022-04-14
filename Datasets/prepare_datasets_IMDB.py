import os
import pandas as pd
import json

from Datasets.Common_Functions import get_last_saved_json, find_tags, Bigram_process

limit = 16000
folder = os.getcwd()

dataset = dict()
dict_to_json = dict()
df = pd.read_csv(folder + "/IMDB Dataset_v1.csv")

sentiment_dict = {
    'negative': 0,
    'positive': 1
}

last_iteration = get_last_saved_json(folder, 'IMDB')
if last_iteration:
    with open(folder + '/IMDB_Dataset_{}.json'.format(last_iteration), 'r') as f:
        dict_to_json = json.loads(f.read())
df = df[last_iteration:limit]
for i, row in df.iterrows():
    dataset.update({
        last_iteration: {
        'original': row['review'],
        'class': sentiment_dict[row['sentiment']]
        }
    })
    last_iteration += 1

find_tags(dataset, dict_to_json, folder, 'IMDB', limit)
Bigram_process(dataset, dict_to_json, folder, 'IMDB', 5000)
