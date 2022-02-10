import numpy as np
import pandas as pd
import json
import stanza
from zipfile import ZipFile

limit = 3000

dict_to_json = dict()
df = pd.read_csv("Datasets/IMDB Dataset_v1.csv")

'''
Stanza documentation - https://stanfordnlp.github.io/stanza/index.html
UPOS documentation - https://universaldependencies.org/u/pos/
'''
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')

for i, row in df.iterrows():
    dict_to_json[i] = {
        'review': row['review'],
        'sentiment': row['sentiment']
    }
    doc = nlp(row['review'])
    dict_to_json[i].update({"upos": " ".join([word.upos for sent in doc.sentences for word in sent.words])})
    dict_to_json[i].update({"xpos": " ".join([word.xpos for sent in doc.sentences for word in sent.words])})
    if i >= limit:
        break

with open('IMDB_Dataset.json', 'w') as f:
    json.dump(dict_to_json, f)

# with ZipFile('Datasets/IMDB_Dataset.zip', 'w') as zipObj:
#     zipObj.write('Datasets/IMDB_Dataset.json')
