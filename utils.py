
import json
import pickle
import six
import collections


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python 3?")


def write_to_json(data, path):

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def read_from_json(path, ordered_dict=False):
    if ordered_dict == False:
        with open(path) as f:
            data = json.load(f)
        return data
    else:
        with open(path) as f:
            data = json.load(f, object_pairs_hook=collections.OrderedDict)
        return data


def read_to_pickle(data, path):

    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def read_from_pickle(path):

    pickle_in = open(path, "rb")
    data = pickle.load(pickle_in)
    return data
