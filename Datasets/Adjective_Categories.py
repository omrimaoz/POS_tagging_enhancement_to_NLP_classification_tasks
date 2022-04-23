import re

ADJ_categories = {
    'quantity': ['few', 'no', 'three', 'several', 'each'],
    'opinion': ['unusual', 'lovely', 'beautiful', 'strange', 'amazing'],
    'personality/emotion': ['happy', 'sad', 'frightened', 'outgoing', 'grumpy'],
    'sound': ['loud', 'soft', 'vociferous', 'thunderous', 'quiet'],
    'taste': ['sweet', 'bitter', 'tasty', 'yummy', 'blasteless'],
    'touch': ['silky', 'smooth', 'grainy', 'scaly', 'polished'],
    'size': ['big', 'small', 'tall', 'long', 'little'],
    'smell': ['acrid','burnt', 'smelly', 'noxious', 'fragrant'],
    'speed': ['fast', 'rushing', 'rapid', 'snappy', 'swift'],
    'temperature': ['hot', 'freezing', 'icy', 'chilly', 'sizzling', 'steaming'],
    'distance': ['short', 'far', 'distant', 'remote', 'neighboring'],
    'miscellaneous qualities': ['full', 'dry', 'open', 'closed' , 'ornate'],
    'brightness': ['dark', 'bright', 'radiant', 'pale', 'dull'],
    'time': ['early', 'morning', 'initial', 'overdue', 'punctual'],
    'shape': ['round', 'square', 'rectangular', 'narrow', 'fat'],
    'age': ['young', 'old', 'youthful', 'childish', 'antique'],
    'colour': ['blue', 'red', 'pink', 'green', 'black-haired'],
    'origin': ['Dutch', 'Japanese', 'Turkish', 'northern', 'oceanic'],
    'material': ['metal', 'wood', 'plastic', 'metallic', 'plastic'],
    'type': ['general-purpose', 'four-sided', 'U-shaped', 'bread-like', 'bleary-eyed'],
    'purpose': ['cleaning', 'hammering', 'cooking', 'riding', 'gardening']
}
ADJ_suffix = {
    'group': [re.compile('([A-Za-z]+\-[A-Za-z]+)$')],
    'capable_of_being': [re.compile('(.+able)$'), re.compile('(.+ible)$')],
    'pertaining_to': [re.compile('(.+ic)$'), re.compile('(.+ical)$')],
    'pertaining_to_al': [re.compile('(.+al)$')],
    'reminiscent_of': [re.compile('(.+esque)$')],
    'notable_for': [re.compile('(.+ful)$')],
    'noun_ly': [re.compile('(.+ly)$')],
    'characterized_by': [re.compile('(.+ious)$'), re.compile('(.+ous)$')],
    'characterized_by_y': [re.compile('(.+y)$')],
    'having_the_quality_of': [re.compile('(.+ish)$')],
    'having_the_nature_of': [re.compile('(.+ive)$')],
    'without': [re.compile('(.+less)$')],
    'placenames': [re.compile('(.+ese)$'), re.compile('(.+i)$'), re.compile('(.+ish)$'), re.compile('(.+ian)$')],
    'emotion': [re.compile('(.+ed)$')],
    'thing_situation': [re.compile('(.+ing)$')],
    'doing_performing': [re.compile('(.+ent)$')]
}
