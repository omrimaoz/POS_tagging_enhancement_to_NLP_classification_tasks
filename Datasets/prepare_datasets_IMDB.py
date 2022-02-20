import os
import re

import numpy as np
import pandas as pd
import json
import stanza
from zipfile import ZipFile

limit = 25000
folder = './Datasets/'

dict_to_json = dict()
df = pd.read_csv("Datasets/IMDB Dataset_v1.csv")

'''
Stanza documentation - https://stanfordnlp.github.io/stanza/index.html
UPOS documentation - https://universaldependencies.org/u/pos/
'''
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')

listdir = os.listdir(folder)
regex = re.compile('_(\d+)\.json')
listdir = [regex.findall(file_name) for file_name in listdir if 'IMDB' in file_name]
listdir = [int(file_name[0]) for file_name in listdir if len(file_name) > 0]
listdir.sort()
if len(listdir) > 0:
    with open(folder + 'IMDB_Dataset_{}.json'.format(listdir[-1]), 'r') as f:
        dict_to_json = json.loads(f.read())
    df = df[listdir[-1]:]

for i, row in df.iterrows():
    if (i+1) % 1000 == 0:
        print('Process {}/{}'.format(i+1, limit))
    dict_to_json[i] = {
        'review': row['review'],
        'sentiment': row['sentiment']
    }
    doc = nlp(row['review'])
    dict_to_json[i].update({"upos": " ".join([word.upos for sent in doc.sentences for word in sent.words])})
    dict_to_json[i].update({"xpos": " ".join([word.xpos for sent in doc.sentences for word in sent.words])})

    if (i+1) % 5000 == 0 or (i+1) >= limit:
        file_name = folder + 'IMDB_Dataset_{}.json'.format(i+1)
        with open(file_name, 'w') as f:
            json.dump(dict_to_json, f)

    if (i+1) >= limit:
        break

# with ZipFile('Datasets/IMDB_Dataset.zip', 'w') as zipObj:
#     zipObj.write('Datasets/IMDB_Dataset.json')
