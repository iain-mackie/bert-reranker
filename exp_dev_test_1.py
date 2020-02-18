
import torch
from bert_models import BertReRanker, fine_tuning_bert_re_ranker
from preprocessing import build_data_loader

if __name__ == "__main__":


    dev_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_dataset_from_pickle_v2.pt'
    test_path = '/nfs/trec_car/data/bert_reranker_datasets/test_dataset_from_pickle_v2.pt'

    print('loading train tensor')
    train_tensor = torch.load(dev_path)
    print('loading dev tensor')
    validation_tensor = torch.load(test_path)
    batch_size = 8

    train_dataloader, validation_dataloader = build_data_loader(train_tensor=train_tensor,
                                                                validation_tensor=validation_tensor,
                                                                batch_size=batch_size)

    print('init Bert')
    pretrained_weights = 'bert-base-uncased'
    relevance_bert = BertReRanker.from_pretrained(pretrained_weights)
    epochs = 5
    lr = 5e-5
    eps = 1e-8
    seed_val = 42
    write = True
    model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    experiment_name = 'exp_dev_test_3'
    do_eval = True
    fine_tuning_bert_re_ranker(model=relevance_bert, train_dataloader=train_dataloader,
                               validation_dataloader=validation_dataloader, epochs=epochs, lr=lr, eps=eps,
                               seed_val=seed_val, write=write, model_path=model_path, experiment_name=experiment_name,
                               do_eval=do_eval)