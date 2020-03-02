
import json
import collections
import pickle


def write_to_json(data, path):

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def write_to_json_given_suffix(data_path, set_name, data, suffix):

    path = data_path + set_name + suffix
    write_to_json(data=data, path=path)


def read_from_json(path, ordered_dict=False):

    if ordered_dict == False:
        with open(path) as f:
            data = json.load(f)
        return data
    else:
        with open(path) as f:
            data = json.load(f, object_pairs_hook=collections.OrderedDict)
        return data


def read_from_json_given_suffix(data_path, set_name, suffix, ordered_dict=False):

    path = data_path + set_name + suffix
    return read_from_json(path=path, ordered_dict=ordered_dict)


def read_to_pickle(data, path):

    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def read_from_pickle(path):

    pickle_in = open(path, "rb")
    data = pickle.load(pickle_in)
    return data
