import os
import json
import random

from Datasets.Feature_Selection import get_last_saved_json, feature_selection

limit = 15000
folder = os.getcwd()
random.seed(0)

proceed_lines = list()
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
category_ids = [0, 1, 2, 3, 9]
chosen_categories = [inv_categories_dict[category_id] for category_id in category_ids]

last_iteration = get_last_saved_json(folder, 'News')
if last_iteration:
    with open(folder + '/News_Dataset_{}.json'.format(last_iteration), 'r') as f:
        dict_to_json = json.loads(f.read())

with open(folder + '/News_Dataset_v1.json', 'r') as f:
    lines = f.readlines()

lines = [line for line in lines if json.loads(line)['category'] in chosen_categories]
category_counter = [0, 0, 0, 0, 0, 0, 0]
for line in lines:
    cat = int(category_ids.index(categories_dict[json.loads(line)['category']]))
    if category_counter[cat] < limit // len(category_ids) + 1:
        category_counter[cat] += 1
        proceed_lines.append(line)

proceed_lines = proceed_lines[last_iteration:limit]
random.shuffle(proceed_lines)

if proceed_lines:
    with open(folder + '/News_Dataset_v2.json', 'w') as f:
        f.write('{')
        for i, line in enumerate(proceed_lines):
            if i == len(proceed_lines) - 1:
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

feature_selection(dataset, dict_to_json, folder, 'News', limit)

