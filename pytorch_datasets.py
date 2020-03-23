
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, TensorDataset
from utils.file_utils import read_from_json, read_from_json_given_suffix

import torch
import random


def get_class_data_inputs(data_path, set_name, c):

    suffixes = [('input_ids', '_input_ids_{}.json'),
                ('token_type_ids', '_token_type_ids_{}.json'),
                ('attention_mask', '_attention_mask_{}.json'),
                ('labels', '_labels_{}.json')]
    class_data_inputs = {}
    for name, s in suffixes:
        data = read_from_json_given_suffix(data_path=data_path, set_name=set_name, suffix=s.format(c), ordered_dict=False)
        class_data_inputs[name] = data
    return class_data_inputs


def convert_training_dataset_to_pt(set_name, data_path, output_path, include_qrels=False, pad_rel_docs=False):

    data_inputs = {}
    for c in ['rel', 'not_rel', 'qrels']:
        data_inputs[c] = get_class_data_inputs(data_path=data_path, set_name=set_name, c=c)

    len_rel = len(data_inputs['rel']['input_ids'])
    len_not_rel = len(data_inputs['not_rel']['input_ids'])
    len_qrels = len(data_inputs['qrels']['input_ids'])

    if include_qrels:
        print('For R docs --> using run ({} docs) + qrels ({} docs)'.format(len_rel, len_qrels))
        input_ids_rel = data_inputs['rel']['input_ids'] + data_inputs['qrels']['input_ids']
        token_type_ids_rel = data_inputs['rel']['token_type_ids'] + data_inputs['qrels']['token_type_ids']
        attention_mask_rel = data_inputs['rel']['attention_mask'] + data_inputs['qrels']['attention_mask']
        labels_rel = data_inputs['rel']['labels'] + data_inputs['qrels']['labels']
    else:
        print('For R docs --> solely run ({} docs)'.format(len_rel))
        input_ids_rel = data_inputs['rel']['input_ids']
        token_type_ids_rel = data_inputs['rel']['token_type_ids']
        attention_mask_rel = data_inputs['rel']['attention_mask']
        labels_rel = data_inputs['rel']['labels']

    if pad_rel_docs:
        diff = len_not_rel - len_rel
        print('not rel ({} docs) minus rel ({} docs) --> add {} sampled docs'.format(len_not_rel, len_rel, diff))
        idx_list = list(range(len(input_ids_rel)))
        for idx in random.choices(idx_list, k=diff):
            input_ids_rel += [input_ids_rel[idx]]
            token_type_ids_rel += [token_type_ids_rel[idx]]
            attention_mask_rel += [attention_mask_rel[idx]]
            labels_rel += [labels_rel[idx]]

    print('Adding {} not rel docs'.format(len_not_rel))
    input_ids = input_ids_rel + data_inputs['not_rel']['input_ids']
    token_type_ids = token_type_ids_rel + data_inputs['not_rel']['token_type_ids']
    attention_mask = attention_mask_rel + data_inputs['not_rel']['attention_mask']
    labels = labels_rel + data_inputs['not_rel']['labels']

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

    data = read_from_json_given_suffix(data_path=data_path, set_name=set_name, suffix=suffix, ordered_dict=False)
    data_tensor = torch.tensor(data)
    print('{} --> tensor.shape: {}, tensor.size: {}'.format(set_name + suffix, data_tensor.shape, data_tensor.size(0)))
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

    for i in ['25', '100']:

        data_path = '/nfs/trec_car/data/bert_reranker_datasets/training_data/'
        set_name = 'train_benchmarkY1_' + i
        output_path = '/nfs/trec_car/data/bert_reranker_datasets/training_data/train_benchmarkY1_{}_dataset_no_qrels_pad.pt'.format(i)
        convert_training_dataset_to_pt(set_name=set_name, data_path=data_path, output_path=output_path, include_qrels=False,
                                       pad_rel_docs=True)


