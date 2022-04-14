import os
import json
import random

from Datasets.Common_Functions import get_last_saved_json, find_tags, Bigram_process

limit = 5000
folder = os.getcwd()
random.seed(0)

dataset = dict()
dict_to_json = dict()
News_dict = dict()

categories_dict = {
    'POLITICS': 0,
    'ENTERTAINMENT': 1,
    'WELLNESS': 2,
    'TRAVEL': 3,
    'PARENTING': 4,
    'STYLE & BEAUTY': 5,
    'HEALTHY LIVING': 6,
    'QUEER VOICES': 7,
    'FOOD & DRINK': 8,
    'BUSINESS': 9,
    'COMEDY': 10,
    'SPORTS': 11,
    'BLACK VOICES': 12,
    'PARENTS': 13,
    'THE WORLDPOST': 14,
    'HOME & LIVING': 15,
    'WOMEN': 16,
    'CRIME': 17,
    'IMPACT': 18,
    'WEDDINGS': 19,
    'DIVORCE': 20,
    'MEDIA': 21,
    'WEIRD NEWS': 22,
    'GREEN': 23,
    'WORLDPOST': 24,
    'RELIGION': 25,
    'STYLE': 26,
    'WORLD NEWS': 27,
    'TASTE': 28,
    'SCIENCE': 29,
    'TECH': 30,
    'ARTS': 31,
    'FIFTY': 32,
    'GOOD NEWS': 33,
    'ARTS & CULTURE': 34,
    'MONEY': 35,
    'COLLEGE': 36,
    'ENVIRONMENT': 37,
    'LATINO VOICES': 38,
    'EDUCATION': 39,
    'CULTURE & ARTS': 40
}
inv_categories_dict = {v: k for k, v in categories_dict.items()}
category_ids = [0, 1, 2, 3, 9, 11, 17]
chosen_categories = [inv_categories_dict[category_id] for category_id in category_ids]

last_iteration = get_last_saved_json(folder, 'News')
if last_iteration:
    with open(folder + '/News_Dataset_{}.json'.format(last_iteration), 'r') as f:
        dict_to_json = json.loads(f.read())

with open(folder + '/News_Dataset_v1.json', 'r') as f:
    lines = f.readlines()
    lines = [line for line in lines if json.loads(line)['category'] in chosen_categories]
    random.shuffle(lines)
    lines = lines[last_iteration:limit]

if lines:
    with open(folder + '/News_Dataset_v2.json', 'w') as f:
        f.write('{')
        lines = lines[:limit]
        for i, line in enumerate(lines):
            if i == len(lines) - 1:
                f.write('"' + str(i) + '":' + line + '}')
                break
            f.write('"' + str(i) + '":' + line + ",")

    with open(folder + '/News_Dataset_v2.json', 'r') as f:
        News_dict = json.loads(f.read())
    os.remove(folder + '/News_Dataset_v2.json')

for key, item in News_dict.items():
    dataset.update({
        key: {
            'original': item['headline'],
            'class': category_ids.index(categories_dict[item['category']])
        }
    })

find_tags(dataset, dict_to_json, folder, 'News', limit)
Bigram_process(dataset, dict_to_json, folder, 'News', 5000)
