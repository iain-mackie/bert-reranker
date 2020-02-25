
from trec_car_tools import iter_paragraphs, ParaText
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from utils import read_from_json, write_to_json, convert_to_unicode

import collections
import time
import torch
import lmdb


# def build_training_dataset(data_path, corpus_path, set_name, tokenizer, max_length=512):
#
#     print('Building {} dataset as jsons'.format(set_name))
#     start_time = time.time()
#
#     print('reading merged data')
#     read_path = data_path + set_name + '_merge.json'
#     data = read_from_json(path=read_path, ordered_dict=True)
#
#     input_ids_rel = []
#     token_type_ids_rel = []
#     attention_mask_rel = []
#     labels_rel = []
#
#     input_ids_not_rel = []
#     token_type_ids_not_rel = []
#     attention_mask_not_rel = []
#     labels_not_rel= []
#
#     with SqliteDict(corpus_path) as mydict:
#
#         for i, query in enumerate(data):
#
#             qrels, doc_titles = data[query]
#             query = query.replace('enwiki:', '')
#             query = query.replace('%20', ' ')
#             query = query.replace('/', ' ')
#             query = convert_to_unicode(query)
#             if i % 1000 == 0:
#                 print('query', query)
#
#             all_docs = list(set(doc_titles + qrels))
#
#             for d in all_docs:
#
#                 text = mydict[d]
#                 q_d = tokenizer.encode_plus(text=query, text_pair=convert_to_unicode(text), max_length=max_length,
#                                             add_special_tokens=True, pad_to_max_length=True)
#
#                 if d in qrels:
#                     input_ids_rel += [q_d['input_ids']]
#                     token_type_ids_rel += [q_d['token_type_ids']]
#                     attention_mask_rel += [q_d['attention_mask']]
#                     labels_rel += [[1]]
#                 else:
#                     input_ids_not_rel += [q_d['input_ids']]
#                     token_type_ids_not_rel += [q_d['token_type_ids']]
#                     attention_mask_not_rel += [q_d['attention_mask']]
#                     labels_not_rel += [[0]]
#
#             if i % 1000 == 0:
#                 print('wrote {}, {} of {} queries'.format(set_name, i, len(data)))
#                 time_passed = time.time() - start_time
#                 est_hours = (len(data) - i) * time_passed / (max(1.0, i) * 3600)
#                 print('estimated total hours to save: {}'.format(est_hours))
#
#     print('len of input_ids: rel {}, not rel: {}'.format(len(input_ids_rel), len(input_ids_not_rel)))
#     print('len of token_type_ids: rel {}, not rel: {}'.format(len(token_type_ids_rel), len(token_type_ids_not_rel)))
#     print('len of attention_mask: rel {}, not rel: {}'.format(len(attention_mask_rel), len(attention_mask_not_rel)))
#     print('len of labels: rel {}, not rel: {}'.format(len(labels_rel), len(labels_not_rel)))
#
#
#
#     # print('Writing lists to jsons')
    # path = data_path + '{}_input_ids.json'.format(set_name)
    # write_to_json(data=input_ids, path=path)
    #
    # path = data_path + '{}_token_type_ids.json'.format(set_name)
    # write_to_json(data=token_type_ids, path=path)
    #
    # path = data_path + '{}_attention_mask.json'.format(set_name)
    # write_to_json(data=attention_mask, path=path)
    #
    # path = data_path + '{}_labels.json'.format(set_name)
    # write_to_json(data=labels, path=path)
    # print('Done')


def build_validation_dataset(data_path, lmdb_path, set_name, tokenizer, max_length=512):

    print('Building {} dataset as jsons'.format(set_name))
    start_time = time.time()

    print('reading merged data')
    read_path = data_path + set_name + '_merge.json'
    data = read_from_json(path=read_path, ordered_dict=True)

    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels = []

    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:

        for i, query in enumerate(data):

            qrels, doc_titles = data[query]
            query = query.replace('enwiki:', '')
            query = query.replace('%20', ' ')
            query = query.replace('/', ' ')
            query = convert_to_unicode(query)
            if i % 1000 == 0:
                print('query', query)

            for d in doc_titles:
                text = txn.get(d.ecode())
                q_d = tokenizer.encode_plus(text=query, text_pair=convert_to_unicode(text), max_length=max_length,
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


def load_corpus(corpus_path, lmdb_path, map_size=1e11):
    """Loads TREC-CAR's paraghaphs into a dict of key: title, value: paragraph."""
    start_time = time.time()
    APPROX_TOTAL_PARAGRAPHS = 30000000
    env = lmdb.open(path=lmdb_path, map_size=map_size)
    with env.begin(write=True) as txn:
        with open(corpus_path, 'rb') as f:
            for i, p in enumerate(iter_paragraphs(f)):
                para_txt = [elem.text if isinstance(elem, ParaText)
                            else elem.anchor_text
                            for elem in p.bodies]
                txn.put(str(p.para_id).encode(), ' '.join(para_txt).encode())
                if i % 1000000 == 0:
                    print(str(p.para_id))
                    print('Loading paragraph {} of {}'.format(i, APPROX_TOTAL_PARAGRAPHS))
                    time_passed = time.time() - start_time
                    hours_remaining = (APPROX_TOTAL_PARAGRAPHS - i) * time_passed / (max(1.0, i) * 3600)
                    print('Estimated hours remaining to load corpus: {}'.format(hours_remaining))


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


def merge(qrels, run):
    """Merge qrels and runs into a single dict of key: topic, value: tuple(relevant_doc_ids, candidate_doc_ids)"""
    data = collections.OrderedDict()
    for topic, candidate_doc_ids in run.items():
        data[topic] = (qrels[topic], candidate_doc_ids)
    return data


def preprocess_runs_and_qrels(set_name, data_path):

    print('building run')
    read_path = data_path + '{}.run'.format(set_name)
    write_path = data_path + set_name + '_run.json'
    run = load_run(path=read_path)
    write_to_json(data=run, path=write_path)

    print('building qrels')
    read_path = data_path + '{}.qrels'.format(set_name)
    write_path = data_path + set_name + '_qrels.json'
    qrels = load_qrels(path=read_path)
    write_to_json(data=qrels, path=write_path)

    print('merging run + qrels')
    write_path = data_path + set_name + '_merge.json'
    data = merge(qrels=qrels, run=run)
    write_to_json(data=data, path=write_path)


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
    corpus_path = '/nfs/trec_car/data/paragraphs/dedup.articles-paragraphs.cbor'
    lmdb_path = '/nfs/trec_car/data/bert_reranker_datasets/trec_car_lmdb'
    load_corpus(corpus_path=corpus_path, lmdb_path=lmdb_path)

    print('preprocessing runs and qrels')
    # will look for {set_name}.run + {set_name}.qrels
    data_dir = '/nfs/trec_car/data/bert_reranker_datasets/'
    set_name = 'test'
    temp_file = True
    is_train = False
    preprocess_runs_and_qrels(set_name=set_name, data_path=data_dir)

    max_length = 512
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if is_train:
        print('building training dataset: {}'.format(set_name))

    else:
        print('building validation / test dataset: {}'.format(set_name))
        build_validation_dataset(data_path=data_dir, lmdb_path=lmdb_path, set_name=set_name, tokenizer=tokenizer,
                                 max_length=max_length)





