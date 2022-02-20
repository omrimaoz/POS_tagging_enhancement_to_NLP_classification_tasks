#imports
import json
import re
import string
import sys
from collections import Counter

import numpy as np
import pandas as pd
import spacy

destination_folder = '../LSTM'
dataset = sys.argv[1]
tagging = sys.argv[2]  # tagging = 'review' / 'upos' / 'xpos'

#loading the data
with open("../Datasets/" + dataset, 'r') as f:
    reviews = json.loads(f.read())
# reviews = reviews[reviews['Review Text'].notna()]
print('Number of reviews: {}'.format(len(reviews)))

#tokenization
tok = spacy.load('en_core_web_sm')
def tokenize (text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]

#count number of occurences of each word
counts = Counter()
for _, review in reviews.items():
    counts.update(tokenize(review['review']))

# deleting infrequent words
print("Number of words before:", len(counts.keys()))
for word in list(counts):
    if counts[word] < 2:
        del counts[word]
print("Number of words after:", len(counts.keys()))

#creating vocabulary
vocab2index = {"": 0, "UNK": 1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)

encode_reviews = {
    'prop': {
        'num_reviews': len(reviews),
        'num_words': len(words)
    },
    'data': {}
}

def encode_sentence(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded.tolist(), length

for idx, review in reviews.items():
    sentiment = 0
    encode_sen, len_sen = encode_sentence(review[tagging], vocab2index)
    if 'IMDB' in dataset:
        sentiment = 1 if review['sentiment'] == 'positive' else 0
    if 'News' in dataset:
        pass  # TODO: fill in
    encode_reviews['data'].update({idx: {
        'encode_sen': encode_sen,
        'len_sen': len_sen,
        'sentiment': sentiment,
    }})

with open('../LSTM/Processed_' + tagging + '_' + dataset, 'w') as f:
    f.write(json.dumps(encode_reviews))
