import numpy as np
import pandas as pd
import json
import stanza
from zipfile import ZipFile

limit = 3000

with open('Datasets/News_Dataset_v1.json', 'r') as f:
    lines = f.readlines()
with open('Datasets/News_Dataset_v2.json', 'w') as f:
    f.write('{')
    lines = lines[:limit]
    for i, line in enumerate(lines):
        if i == len(lines) - 1:
            f.write('"' + str(i) + '":' + line + '}')
            break
        f.write('"' + str(i) + '":' + line + ",")

dict_to_json = dict()
with open('Datasets/News_Dataset_v2.json', 'r') as f:
    News_dict = json.loads(f.read())

'''
Stanza documentation - https://stanfordnlp.github.io/stanza/index.html
UPOS documentation - https://universaldependencies.org/u/pos/
'''
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')

for key in News_dict.keys():
    doc = nlp(News_dict[key]['headline'])
    News_dict[key].update({"upos": " ".join([word.upos for sent in doc.sentences for word in sent.words])})
    News_dict[key].update({"xpos": " ".join([word.xpos for sent in doc.sentences for word in sent.words])})

with open('Datasets/News_Dataset.json', 'w') as f:
    json.dump(News_dict, f)

# with ZipFile('Datasets/News_Dataset.zip', 'w') as zipObj:
#     zipObj.write('Datasets/News_Dataset.json')
