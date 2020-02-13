
from transformers import BertModel, BertTokenizer, BertPreTrainedModel
from torch.utils.data import TensorDataset
import torch

from bert_retrieval_model import BertForRelevance
import os

# BERT init
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

queries = ["find me a good match for my query, please"]
docs = [("Great query I am", 1.0),
        ("Yup, I'm a rubbish query", 0.0),
        ("Okay query for sure", 1.0),
        ("Who knows", 0.0)
        ]

train_inputs = []
train_labels = []
for q in queries:
    for d in docs:
        q_d_pair = tokenizer.encode(q, d[0], max_length=512, add_special_tokens=True, pad_to_max_length=True)
        q_d_pair_plus = tokenizer.encode_plus(q, d[0], max_length=512, add_special_tokens=True, pad_to_max_length=True)
        # should use - be explicit!
        # print(q_d_pair_plus['input_ids'])
        # print(q_d_pair_plus['token_type_ids'])
        # print(q_d_pair_plus['attention_mask'])

        train_inputs.append(q_d_pair)
        train_labels.append(d[1])

#train_inputs_tensor = torch.tensor(q_d_pair_plus['input_ids'], q_d_pair_plus['token_type_ids'], q_d_pair_plus['attention_mask'])
train_inputs_tensor = torch.tensor(q_d_pair)
train_labels_tensor = torch.tensor(train_labels)

if __name__ == "__main__":

    print(train_inputs_tensor)

    from fine_tunning import build_data_loader, train_bert_relevance_model
    #
    both_tensor = TensorDataset(train_inputs_tensor, train_labels_tensor)
    #
    # train_dataloader, validation_dataloader = build_data_loader(train_tensor=both_tensor,
    #                                                             validation_tensor=both_tensor,
    #                                                             batch_size=2)
    #
    # relevance_bert = BertForRelevance.from_pretrained(pretrained_weights)

    # train_bert_relevance_model(model=relevance_bert,
    #                            train_dataloader=train_dataloader,
    #                            validation_dataloader=validation_dataloader,
    #                            epochs=2,
    #                            lr=5e-4,
    #                            eps=1e-8)

    # outputs = relevance_bert(train_inputs_tensor, labels=train_labels_tensor)
    #
    #
    #
    #
    #
    #
    #
    #
    #
