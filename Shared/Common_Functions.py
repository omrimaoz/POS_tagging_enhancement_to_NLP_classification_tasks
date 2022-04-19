import re
import string


def remove_punctuation(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(
        string.punctuation.replace('_', '').replace('#', '')) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
    nopunct = regex.sub("", text.lower())
    return nopunct