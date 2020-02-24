
from trec_car_tools import iter_paragraphs, ParaText
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import collections
import time
import six
import unicodedata
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import torch
import pickle
import json
from sqlitedict import SqliteDict

#TODO - make datasets (paragraph + sentence)
#TODO - sense check


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


def read_from_json(path):

    with open(path) as f:
        data = json.load(f)
    return data


def read_to_pickle(data, path):

    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def read_from_pickle(path):

    pickle_in = open(path, "rb")
    data = pickle.load(pickle_in)
    return data


def build_dataset(data, corpus, set_name, tokenizer, data_path=None, max_length=512):

    print('Building {} dataset as pickles'.format(set_name))
    start_time = time.time()

    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels = []

    for i, query in enumerate(data):

        qrels, doc_titles = data[query]
        query = query.replace('enwiki:', '')
        query = query.replace('%20', ' ')
        query = query.replace('/', ' ')
        query = convert_to_unicode(query)
        if i % 1000 == 0:
            print('query', query)

        for d in doc_titles:
            q_d = tokenizer.encode_plus(text=query, text_pair=convert_to_unicode(corpus[d]), max_length=max_length,
                                        add_special_tokens=True, pad_to_max_length=True)
            input_ids += [q_d['input_ids']]
            token_type_ids += [q_d['token_type_ids']]
            attention_mask += [q_d['attention_mask']]

            if d in qrels:
                labels += [[1]]
            else:
                labels += [[0]]

        if i % 1000 == 0:
            print('wrote {}, {} of {} queries'.format(set_name, i, len(data)))
            time_passed = time.time() - start_time
            est_hours = (len(data) - i) * time_passed / (max(1.0, i) * 3600)
            print('estimated total hours to save: {}'.format(est_hours))

    print('len of input_ids: {}'.format(len(input_ids)))
    print('len of token_type_ids: {}'.format(len(token_type_ids)))
    print('len of attention_mask: {}'.format(len(attention_mask)))
    print('len of labels: {}'.format(len(labels)))

    print('Writing lists to jsons')
    path = data_path + '{}_input_ids.json'.format(set_name)
    write_to_json(data=input_ids, path=path)

    path = data_path + '{}_token_type_ids.json'.format(set_name)
    write_to_json(data=token_type_ids, path=path)

    path = data_path + '{}_attention_mask.json'.format(set_name)
    write_to_json(data=attention_mask, path=path)

    path = data_path + '{}_labels.json'.format(set_name)
    write_to_json(data=labels, path=path)
    print('Done')


def convert_dataset_to_pt(set_name, data_path, output_path):

    path = data_path + '{}_input_ids.json'.format(set_name)
    print('reading file: {}'.format(path))
    input_ids = read_from_json(path=path)

    path = data_path + '{}_token_type_ids.json'.format(set_name)
    print('reading file: {}'.format(path))
    token_type_ids = read_from_json(path=path)

    path = data_path + '{}_attention_mask.json'.format(set_name)
    print('reading file: {}'.format(path))
    attention_mask = read_from_json(path=path)

    path = data_path + '{}_labels.json'.format(set_name)
    print('reading file: {}'.format(path))
    labels = read_from_json(path=path)

    input_ids_tensor = torch.tensor(input_ids)
    token_type_ids_tensor = torch.tensor(token_type_ids)
    attention_mask_tensor = torch.tensor(attention_mask)
    labels_tensor = torch.tensor(labels)

    print('tensor shape of input_ids: {}'.format(input_ids_tensor.shape))
    print('tensor shape token_type_ids: {}'.format(token_type_ids_tensor.shape))
    print('tensor shape attention_mask: {}'.format(attention_mask_tensor.shape))
    print('tensor shape labels: {}'.format(labels_tensor.shape))

    dataset = TensorDataset(input_ids_tensor, token_type_ids_tensor, attention_mask_tensor, labels_tensor)

    print('saving tensor to: {}'.format(output_path))
    torch.save(dataset, output_path)


def load_qrels(path):
    """Loads qrels into a dict of key: topic, value: list of relevant doc ids."""
    qrels = collections.defaultdict(list)
    with open(path) as f:
        for i, line in enumerate(f):
            topic, _, doc_title, relevance = line.rstrip().split(' ')
            if int(relevance) >= 1:
                qrels[topic].append(doc_title)
            if i % 1000000 == 0:
                print('Loading qrels {}'.format(i))
    return qrels


def load_run(path):
    """Loads run into a dict of key: topic, value: list of candidate doc ids."""

    # We want to preserve the order of runs so we can pair the run file with the
    # TFRecord file.
    run = collections.OrderedDict()
    with open(path) as f:
        for i, line in enumerate(f):
            topic, _, doc_title, rank, _, _ = line.split(' ')
            if topic not in run:
                run[topic] = []
            run[topic].append((doc_title, int(rank)))
            if i % 1000000 == 0:
                print('Loading run {}'.format(i))

    # Sort candidate docs by rank.
    sorted_run = collections.OrderedDict()
    for topic, doc_titles_ranks in run.items():
        sorted(doc_titles_ranks, key=lambda x: x[1])
        doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
        sorted_run[topic] = doc_titles

    return sorted_run


def load_corpus(read_path, write_path):
    """Loads TREC-CAR's paraghaphs into a dict of key: title, value: paragraph."""
    start_time = time.time()
    APPROX_TOTAL_PARAGRAPHS = 30000000
    mydict = SqliteDict(write_path, autocommit=True)
    counter = 0
    with open(read_path, 'rb') as f:
        for i, p in enumerate(iter_paragraphs(f)):
            if counter > 10:
                break
            para_txt = [elem.text if isinstance(elem, ParaText)
                        else elem.anchor_text
                        for elem in p.bodies]
            mydict[p.para_id] = ' '.join(para_txt)
            print(p.para_id)
            print(' '.join(para_txt))
            if i % 1000000 == 0:
                print('Loading paragraph {} of {}'.format(i, APPROX_TOTAL_PARAGRAPHS))
                time_passed = time.time() - start_time
                hours_remaining = (APPROX_TOTAL_PARAGRAPHS - i) * time_passed / (max(1.0, i) * 3600)
                print('Estimated hours remaining to load corpus: {}'.format(hours_remaining))
            counter += 1


def merge(qrels, run):
    """Merge qrels and runs into a single dict of key: topic, value: tuple(relevant_doc_ids, candidate_doc_ids)"""
    data = collections.OrderedDict()
    for topic, candidate_doc_ids in run.items():
        data[topic] = (qrels[topic], candidate_doc_ids)
    return data


def preprocess_runs_and_qrels(set_name, data_path, temp_file=True):

    print('building run')
    run_path = data_path + '{}.run'.format(set_name)
    run = load_run(path=run_path)
    if temp_file:
        path = data_path + set_name + '_run.json'
        write_to_json(data=run, path=path)

    print('building qrels')
    qrels_path = data_path + '{}.qrels'.format(set_name)
    qrels = load_qrels(path=qrels_path)
    if temp_file:
        path = data_path + set_name + '_qrels.json'
        write_to_json(data=qrels, path=path)

    print('merging run + qrels')
    data = merge(qrels=qrels, run=run)
    if temp_file:
        path = data_path + set_name + '_merge.json'
        write_to_json(data=data, path=path)


def build_trainig_data_loader(tensor, batch_size):
    print('building training data loader')
    train_sampler = RandomSampler(tensor)
    return DataLoader(tensor, sampler=train_sampler, batch_size=batch_size)


def build_validation_data_loader(tensor, batch_size):
    print('building training data loader')
    validation_sampler = SequentialSampler(tensor)
    return DataLoader(tensor, sampler=validation_sampler, batch_size=batch_size)

if __name__ == "__main__":

    print('build corpus DB')
    read_path = '/nfs/trec_car/data/paragraphs/dedup.articles-paragraphs.cbor'
    write_path = '/nfs/trec_car/index/anserini_paragraphs/lucene-index.car17v2.0.paragraphsv2.sqlite'
    load_corpus(read_path=read_path, write_path=write_path)

    print('preprocessing runs and qrels')
    data_dir = '/nfs/trec_car/data/bert_reranker_datasets/'
    set_name = 'test'
    temp_file = True
    preprocess_runs_and_qrels(set_name=set_name, data_path=data_dir, temp_file=True)




