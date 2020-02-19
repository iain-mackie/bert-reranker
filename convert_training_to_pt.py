
from preprocessing import convert_dataset_to_pt

if __name__ == '__main__':

    output_path = '/nfs/trec_car/data/bert_reranker_datasets/train_1000000+_dataset_from_pickle_v2'
    data_dir = '/nfs/trec_car/data/bert_reranker_datasets/'
    set_name = 'train_1000000+_'

    convert_dataset_to_pt(set_name, data_path=data_dir, output_path=output_path)