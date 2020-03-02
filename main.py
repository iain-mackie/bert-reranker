
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import torch
from bert_models import BertReRanker, fine_tuning_bert_re_ranker, inference_bert_re_ranker
from pytorch_datasets import build_training_data_loader, build_validation_data_loader
import os
import time

# BERT init
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

# Fake data
queries = ["find me a good match for my query, please"]
docs = [("Great query I am", 1.0),
        ("Yup, I'm a rubbish query", 0.0),
        ("Okay query for sure", 0.0),
        ("no idea", 1.0),
        ("please help!!! I have no idea", 0.0),
        ("Who knows", 0.0),
        ("Okay query for sure", 1.0),
        ("Awesome friggin awesome", 1.0)
        ]

# Build fake dataset
input_ids = []
token_type_ids = []
attention_mask = []
labels = []
for q in queries:
    for d in docs:
        #q_d_pair = tokenizer.encode(q, d[0], max_length=512, add_special_tokens=True, pad_to_max_length=True)
        q_d_pair_plus = tokenizer.encode_plus(q, d[0], max_length=512, add_special_tokens=True, pad_to_max_length=True)

        input_ids.append(q_d_pair_plus['input_ids'])
        token_type_ids.append(q_d_pair_plus['token_type_ids'])
        attention_mask.append(q_d_pair_plus['attention_mask'])
        labels.append(d[1])

input_ids_tensor = torch.tensor(input_ids)
token_type_ids_tensor = torch.tensor(token_type_ids)
attention_mask_tensor = torch.tensor(attention_mask)
labels_tensor = torch.tensor(labels)


if __name__ == "__main__":

    #Construct data loaders
    both_tensor = TensorDataset(input_ids_tensor, token_type_ids_tensor, attention_mask_tensor, labels_tensor)
    validation_dataloader = build_validation_data_loader(tensor=both_tensor, batch_size=2)

    # Init Bert Re-Ranker
    relevance_bert = torch.nn.DataParallel(BertReRanker.from_pretrained(pretrained_weights))
    #relevance_bert = BertReRanker.from_pretrained(pretrained_weights)

    # Train & validation run
    exp_dir = os.path.join(os.getcwd(), 'models/')
    experiment_name = 'test_model_20'
    write = True
    do_eval = True
    run_path = os.path.join(os.getcwd(), 'test_data', 'test_model.run')
    qrels_path = os.path.join(os.getcwd(), 'test_data', 'test_model.qrels')

    fine_tuning_bert_re_ranker(model=relevance_bert, train_dataloader=validation_dataloader,
                               validation_dataloader=validation_dataloader, epochs=1, lr=5e-5, eps=1e-8, write=write,
                               experiment_name=experiment_name, exp_dir=exp_dir, do_eval=do_eval, logging_steps=1,
                               run_path=run_path, qrels_path=qrels_path)

    # #Run Inference
    # model_path = os.path.join(os.getcwd(), 'models', 'test_model_10', 'epoch1')
    # write_path = os.path.join(os.getcwd(), 'models', 'test_model_10', 'bert_epoch1.run')
    #model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/exp_toy_large_2/epoch7/'
    #write_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/exp_toy_large_2/bert_epoch7.run'

    # inference_bert_re_ranker(model_path=model_path, dataloader=validation_dataloader, run_path=run_path,
    #                          write_path=write_path, qrels_path=qrels_path)


