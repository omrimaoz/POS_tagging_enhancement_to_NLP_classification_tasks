import re

ADV_categories = {
    'time': ['never', 'lately', 'always', 'during', 'usually'],
    'manner': ['neatly', 'quickly', 'sadly', 'politely', 'lazily'],
    'degree': ['almost', 'too', 'just', 'hardly', 'simply'],
    'place': ['here', 'there', 'inside', 'below', 'everywhere'],
    'probability': ['impossibly', 'surely', 'probably', 'certainly', 'unlikely']
}
ADV_suffix = {
    'manner_ly': [re.compile('(.+ly)$')],
    'direction_ward': [re.compile('(.+ward)$'), re.compile('(.+wards)$')],
    'direction_way': [re.compile('(.+way)$'), re.compile('(.+ways)$')],
    'manner_wises': [re.compile('(.+wise)$')],
    'full_of': [re.compile('(.+full)$'), re.compile('(.+fully)$')]
}
