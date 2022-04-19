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

def encode_dataset(phrases, tags):
    print('Number of phrases: {}'.format(len(phrases)))

    #tokenization
    tok = spacy.load('en_core_web_sm')
    def tokenize(text):
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        regex = re.compile('[' + re.escape(string.punctuation.replace('_', '').replace('#', '')) + '0-9\\r\\t\\n]') # remove punctuation and numbers
        nopunct = regex.sub(" ", text.lower())
        return [token.text for token in tok.tokenizer(nopunct)]

    #count number of occurences of each word
    counts = Counter()
    for _, phrase in phrases.items():
        counts.update(tokenize(phrase[tags]))

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

    encode_phrases = {
        'prop': {
            'num_phrases': len(phrases),
            'num_words': len(words),
            'num_classes': len(set([phrase['class'] for _, phrase in phrases.items()]))
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

    for idx, phrase in phrases.items():
        encode_sen, len_sen = encode_sentence(phrase[tags], vocab2index)
        encode_phrases['data'].update({idx: {
            'encode_sen': encode_sen,
            'len_sen': len_sen,
            'class': phrase['class'],
        }})

    return encode_phrases
