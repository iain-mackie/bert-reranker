
import torch
from bert_models import BertReRanker, fine_tuning_bert_re_ranker
from trec_car_preprocessing import build_data_loader

if __name__ == "__main__":


    train_path = '/nfs/trec_car/data/bert_reranker_datasets/toy_train_dataset.pt'
    dev_path = '/nfs/trec_car/data/bert_reranker_datasets/toy_dev_dataset.pt'

    print('loading train tensor: {}'.format(train_path))
    train_tensor = torch.load(train_path)
    print('loading dev tensor: {}'.format(dev_path))
    validation_tensor = torch.load(dev_path)
    batch_size = 8

    train_dataloader, validation_dataloader = build_data_loader(train_tensor=train_tensor,
                                                                validation_tensor=validation_tensor,
                                                                batch_size=batch_size)

    print('init Bert')
    pretrained_weights = 'bert-base-uncased'
    relevance_bert = BertReRanker.from_pretrained(pretrained_weights)
    epochs = 10
    lr = 5e-5
    eps = 1e-8
    seed_val = 42
    write = True
    model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    experiment_name = 'exp_toy_3'
    do_eval = True
    logging_steps = 20
    num_rank = 10
    fine_tuning_bert_re_ranker(model=relevance_bert, train_dataloader=train_dataloader,
                               validation_dataloader=validation_dataloader, epochs=epochs, lr=lr, eps=eps,
                               seed_val=seed_val, write=write, model_path=model_path, experiment_name=experiment_name,
                               do_eval=do_eval, logging_steps=logging_steps, num_rank=num_rank)
