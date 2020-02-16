
from trec_car_tools import iter_paragraphs, ParaText # ParaLink iter_pages, iter_annotations,
import os
import collections
import time
import six
import unicodedata
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import torch

#TODO - make datasets (paragraph + sentence)


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


def convert_dataset(data, corpus, set_name, tokenizer, output_path, max_length=512):

    print('Converting {} to pytorch dataset'.format(set_name))
    start_time = time.time()

    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels = []
    for i, query in enumerate(data):

        try:
            qrels, doc_titles = data[query]
            query = query.replace('enwiki:', '')
            query = query.replace('%20', ' ')
            query = query.replace('/', ' ')
            query = convert_to_unicode(query)
            if i % 1000 == 0:
                print('query', query)

            for d in doc_titles:
                q_d = tokenizer.encode_plus(
                    text=query,
                    text_pair=convert_to_unicode(corpus[d]),
                    max_length=max_length,
                    add_special_tokens=True,
                    pad_to_max_length=True
                )
                input_ids = q_d['input_ids']
                token_type_ids = q_d['token_type_ids']
                attention_mask = q_d['attention_mask']
                labels += [[1] if doc_title in qrels else [0] for doc_title in doc_titles]

            if i % 1000 == 0:
                print('wrote {}, {} of {} queries'.format(set_name, i, len(data)))
                time_passed = time.time() - start_time
                est_hours = (len(data) - i) * time_passed / (max(1.0, i) * 3600)
                print('estimated total hours to save: {}'.format(est_hours))
        except:
            print('Exception ')

    input_ids_tensor = torch.tensor(input_ids)
    token_type_ids_tensor = torch.tensor(token_type_ids)
    attention_mask_tensor = torch.tensor(attention_mask)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_ids_tensor, token_type_ids_tensor, attention_mask_tensor, labels_tensor)

    print('saving tensor')
    torch.save(dataset, output_path)


def load_qrels(path):
    """Loads qrels into a dict of key: topic, value: list of relevant doc ids."""
    qrels = collections.defaultdict(set)
    with open(path) as f:
        for i, line in enumerate(f):
            topic, _, doc_title, relevance = line.rstrip().split(' ')
            if int(relevance) >= 1:
                qrels[topic].add(doc_title)
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


def load_corpus(path):
    """Loads TREC-CAR's paraghaphs into a dict of key: title, value: paragraph."""
    corpus = {}
    start_time = time.time()
    APPROX_TOTAL_PARAGRAPHS = 30000000
    with open(path, 'rb') as f:
        for i, p in enumerate(iter_paragraphs(f)):
            para_txt = [elem.text if isinstance(elem, ParaText)
                        else elem.anchor_text
                        for elem in p.bodies]

            corpus[p.para_id] = ' '.join(para_txt)
            if i % 1000000 == 0:
                print('Loading paragraph {} of {}'.format(i, APPROX_TOTAL_PARAGRAPHS))
                time_passed = time.time() - start_time
                hours_remaining = (APPROX_TOTAL_PARAGRAPHS - i) * time_passed / (max(1.0, i) * 3600)
                print('Estimated hours remaining to load corpus: {}'.format(hours_remaining))

    return corpus


def merge(qrels, run):
    """Merge qrels and runs into a single dict of key: topic, value: tuple(relevant_doc_ids, candidate_doc_ids)"""
    data = collections.OrderedDict()
    for topic, candidate_doc_ids in run.items():
        data[topic] = (qrels[topic], candidate_doc_ids)
    return data


def make_tensor_dataset(corpus, set_name, tokenizer, data_path, output_path, max_length=512):

    run_path = data_path + '{}.run'.format(set_name)
    run = load_run(path=run_path)

    qrels_path = data_path + '{}.qrels'.format(set_name)
    qrels = load_qrels(path=qrels_path)

    data = merge(qrels=qrels, run=run)

    convert_dataset(data=data, corpus=corpus, set_name=set_name, tokenizer=tokenizer, output_path=output_path,
                    max_length=max_length)


if __name__ == "__main__":

    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

    paragraphs = '/nfs/trec_car/data/paragraphs/dedup.articles-paragraphs.cbor'
    corpus = load_corpus(paragraphs)

    data_dir = '/nfs/trec_car/data/bert_reranker_datasets/'

    max_length = 512

    set_name = 'test'
    output_path =  '/nfs/trec_car/data/bert_reranker_datasets/test_dataset_explicit.pt'
    make_tensor_dataset(corpus, set_name, tokenizer, data_path=data_dir, output_path=output_path, max_length=max_length)
    #
    # set_name = 'dev'
    #output_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_dataset_explicit.pt'
    # make_tensor_dataset(corpus, set_name, tokenizer, data_path=data_dir, output_path=output_path, max_length=max_length)

    #set_name = 'train'
    #output_path = '/nfs/trec_car/data/bert_reranker_datasets/train_dataset_explicit.pt'
    #make_tensor_dataset(corpus, set_name, tokenizer, data_path=data_dir, output_path=output_path, max_length=max_length)

