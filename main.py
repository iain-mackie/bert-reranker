
from transformers import BertModel, BertTokenizer, BertPreTrainedModel

import torch

from bert_relevance_model import BertForRelevance
import os

# BERT init
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

queries = ["find me a good match for my query, please"]
docs = [("Great query I am", 1.0), ("Yup, I'm a rubbish query", 0.0)]

train_inputs = []
train_labels = []
for q in queries:
    for d in docs:
        q_d_pair = tokenizer.encode(q, d[0], max_length=512, add_special_tokens=True, pad_to_max_length=True)
        q_d_pair_plus = tokenizer.encode_plus(q, d[0], max_length=512, add_special_tokens=True, pad_to_max_length=True)
        # should use - be explicit!
        print(q_d_pair)
        print(q_d_pair_plus)

        train_inputs.append(q_d_pair)
        train_labels.append(d[1])

train_inputs_tensor = torch.tensor(train_inputs)
train_labels_tensor = torch.tensor(train_labels)

if __name__ == "__main__":

    from fine_tunning import build_data_loader, train_bert_relevance_model

    train_dataloader, validation_dataloader = build_data_loader(train_inputs=train_inputs_tensor,
                                                                train_labels=train_labels_tensor,
                                                                validation_inputs=train_inputs_tensor,
                                                                validation_labels=train_labels_tensor,
                                                                batch_size=1)

    relevance_bert = BertForRelevance.from_pretrained(pretrained_weights)

    train_bert_relevance_model(model=relevance_bert,
                               train_dataloader=train_dataloader,
                               validation_dataloader=validation_dataloader,
                               epochs=2,
                               lr=5e-4,
                               eps=1e-8,
                               seed=None)

    #outputs = relevance_bert(train_inputs_tensor, labels=train_labels_tensor)









