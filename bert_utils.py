
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, TensorDataset
from preprocessing_utils import read_from_json

import datetime
import os
import itertools
import torch
import random
#TODO - add data shuffle


def get_query_docids_map(run_path):

    query_docids_map = []
    with open(run_path) as ref_file:

        for line in ref_file:
            query, _, doc_id, _, _, _ = line.strip().split(" ")

            query_docids_map.append((query, doc_id))

    return query_docids_map


def get_query_rel_doc_map(qrels_path):
    query_rel_doc_map = {}
    with open(qrels_path) as qrels_file:

        for line in qrels_file:
            query, _, doc_id, _ = line.strip().split(" ")

            if query in query_rel_doc_map:
                query_rel_doc_map[query].append(doc_id)
            else:
                query_rel_doc_map[query] = [doc_id]

    return query_rel_doc_map


def format_time(elapsed):
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def flatten_list(l):
    return list(itertools.chain(*l))


def convert_training_dataset_to_pt(set_name, data_path, output_path, percent_rel=None):
    path = data_path + '{}_input_ids_rel.json'.format(set_name)
    print('reading file: {}'.format(path))
    input_ids_rel = read_from_json(path=path)
    path = data_path + '{}_token_type_ids_rel.json'.format(set_name)
    print('reading file: {}'.format(path))
    token_type_ids_rel = read_from_json(path=path)
    path = data_path + '{}_attention_mask_rel.json'.format(set_name)
    print('reading file: {}'.format(path))
    attention_mask_rel = read_from_json(path=path)
    path = data_path + '{}_labels_rel.json'.format(set_name)
    print('reading file: {}'.format(path))
    labels_rel = read_from_json(path=path)

    path = data_path + '{}_input_ids_not_rel.json'.format(set_name)
    print('reading file: {}'.format(path))
    input_ids_not_rel = read_from_json(path=path)
    path = data_path + '{}_token_type_ids_not_rel.json'.format(set_name)
    print('reading file: {}'.format(path))
    token_type_ids_not_rel = read_from_json(path=path)
    path = data_path + '{}_attention_mask_not_rel.json'.format(set_name)
    print('reading file: {}'.format(path))
    attention_mask_not_rel = read_from_json(path=path)
    path = data_path + '{}_labels_not_rel.json'.format(set_name)
    print('reading file: {}'.format(path))
    labels_not_rel = read_from_json(path=path)

    if isinstance(percent_rel, float):
        print('balancing classes so {} of data is relevent'.format(percent_rel))
        if (percent_rel > 1.0) or (percent_rel < 0.0):
            print('NOT VALID')

        num_rel = len(input_ids_rel)
        num_not_rel = len(input_ids_not_rel)
        percent_rel_current = num_rel / (num_rel + num_not_rel)

        if percent_rel_current > percent_rel:
            print('percent_rel_current > percent_rel -> reduce relevent docs')

            required_reduced = int((num_rel - (percent_rel * num_rel)) / percent_rel)
            print('keeping {} not rel docs ||| reducing rel docs from {} -> {}'.format(num_not_rel, num_rel, required_reduced))
            input_ids_rel_reduced = []
            token_type_ids_rel_reduced = []
            attention_mask_rel_reduced = []
            labels_rel_reduced = []

            random_ids = list(range(num_rel))
            random.shuffle(random_ids)

            for i in random_ids:

                if len(input_ids_rel_reduced) >= required_reduced:
                    break

                input_ids_rel_reduced += [input_ids_rel[i]]
                token_type_ids_rel_reduced += [token_type_ids_rel[i]]
                attention_mask_rel_reduced += [attention_mask_rel[i]]
                labels_rel_reduced += [labels_rel[i]]


            input_ids = input_ids_rel_reduced + input_ids_not_rel
            token_type_ids = token_type_ids_rel_reduced + token_type_ids_not_rel
            attention_mask = attention_mask_rel_reduced + attention_mask_not_rel
            labels = labels_rel_reduced + labels_not_rel


        elif percent_rel_current < percent_rel:
            print('percent_rel_current < percent_rel -> reduce not relevent docs')

            required_reduced = int((num_rel - (percent_rel * num_rel)) / percent_rel)
            print('keeping {} rel docs ||| reducing not rel docs from {} -> {}'.format(num_rel, num_not_rel, required_reduced))

            input_ids_not_rel_reduced = []
            token_type_ids_not_rel_reduced = []
            attention_mask_not_rel_reduced = []
            labels_not_rel_reduced = []

            random_ids = list(range(num_not_rel))
            random.shuffle(random_ids)

            for i in random_ids:

                if len(input_ids_not_rel_reduced) >= required_reduced:
                    break

                input_ids_not_rel_reduced += [input_ids_not_rel[i]]
                token_type_ids_not_rel_reduced += [token_type_ids_not_rel[i]]
                attention_mask_not_rel_reduced += [attention_mask_not_rel[i]]
                labels_not_rel_reduced += [labels_not_rel[i]]

            input_ids = input_ids_rel + input_ids_not_rel_reduced
            token_type_ids = token_type_ids_rel + token_type_ids_not_rel_reduced
            attention_mask = attention_mask_rel + attention_mask_not_rel_reduced
            labels = labels_rel + labels_not_rel_reduced

        else:
            print(' percent_current == percent_rel')
            input_ids = input_ids_rel + input_ids_not_rel
            token_type_ids = token_type_ids_rel + token_type_ids_not_rel
            attention_mask = attention_mask_rel + attention_mask_not_rel
            labels = labels_rel + labels_not_rel

    else:
        print('Adding all training data to dataset')

        input_ids = input_ids_rel + input_ids_not_rel
        token_type_ids = token_type_ids_rel + token_type_ids_not_rel
        attention_mask = attention_mask_rel + attention_mask_not_rel
        labels = labels_rel + labels_not_rel

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

def get_tensor(set_name, data_path, suffix):

    file_name = set_name + suffix
    path = data_path + file_name
    data = read_from_json(path=path)
    data_tensor = torch.tensor(data)
    print('tensor shape of {}: {}'.format(file_name, data_tensor.shape))
    return data_path


def convert_validation_dataset_to_pt(set_name, data_path, output_path):

    input_ids_tensor = get_tensor(set_name=set_name, data_path=data_path, suffix='_input_ids.json')
    token_type_ids_tensor = get_tensor(set_name=set_name, data_path=data_path, suffix='_token_type_ids.json')
    attention_mask_tensor = get_tensor(set_name=set_name, data_path=data_path, suffix='_attention_mask.json')
    labels_tensor = get_tensor(set_name=set_name, data_path=data_path, suffix='_labels.json')

    dataset = TensorDataset(input_ids_tensor, token_type_ids_tensor, attention_mask_tensor, labels_tensor)

    print('saving tensor to: {}'.format(output_path))
    torch.save(dataset, output_path)


def build_training_data_loader(tensor, batch_size):
    print('building training data loader')
    train_sampler = RandomSampler(tensor)
    return DataLoader(tensor, sampler=train_sampler, batch_size=batch_size)


def build_validation_data_loader(tensor, batch_size):
    print('building training data loader')
    validation_sampler = SequentialSampler(tensor)
    return DataLoader(tensor, sampler=validation_sampler, batch_size=batch_size)


if __name__ == '__main__':

    # set_name = 'test'
    # data_path = '/nfs/trec_car/data/bert_reranker_datasets/'
    # output_path = '/nfs/trec_car/data/bert_reranker_datasets/test_dataset.pt'
    # convert_validation_dataset_to_pt(set_name=set_name, data_path=data_path, output_path=output_path)
    qrels_path = os.path.join(os.getcwd(), 'test_model.qrels')
    get_query_rel_doc_map(qrels_path=qrels_path)

