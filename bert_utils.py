
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, TensorDataset
from preprocessing_utils import read_from_json

import datetime
import os
import itertools
import torch


def get_query_docids_map(set_name, data_path):

    run_path = os.path.join(data_path, set_name + ".run")

    query_docids_map = []
    with open(run_path) as ref_file:

        for line in ref_file:
            query, _, doc_id, _, _, _ = line.strip().split(" ")

            query_docids_map.append((query, doc_id))

    return query_docids_map


def format_time(elapsed):
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def flatten_list(l):
    return list(itertools.chain(*l))


def convert_training_dataset_to_pt(set_name, data_path, output_path, percent_train=None):

    if isinstance(percent_train, float):

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

    else:
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


def convert_validation_dataset_to_pt(set_name, data_path, output_path):

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


def build_training_data_loader(tensor, batch_size):
    print('building training data loader')
    train_sampler = RandomSampler(tensor)
    return DataLoader(tensor, sampler=train_sampler, batch_size=batch_size)


def build_validation_data_loader(tensor, batch_size):
    print('building training data loader')
    validation_sampler = SequentialSampler(tensor)
    return DataLoader(tensor, sampler=validation_sampler, batch_size=batch_size)


if __name__ == '__main__':

    set_name = 'train_benchmarkY1'
    data_path = '/nfs/trec_car/data/bert_reranker_datasets/'
    output_path = '/nfs/trec_car/data/bert_reranker_datasets/train_benchmarkY1_None.pt'
    percent_train = None
    convert_training_dataset_to_pt(set_name=set_name, data_path=data_path, output_path=output_path, percent_train=percent_train)

