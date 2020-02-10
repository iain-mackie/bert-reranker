
from transformers import BertModel, BertTokenizer, BertPreTrainedModel

import torch

from head_model import BertForRelevance
import os

# BERT init
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

queries = ["find me a good match for my query, please"]

docs = [("Great query I am", 1), ("Yup, I'm a rubbish query", 0)]

train_inputs = []
train_labels = []
for q in queries:
    for d in docs:
        q_d_pair = tokenizer.encode(q, d[0], max_length=512, add_special_tokens=True, pad_to_max_length=True)
        print(q_d_pair)
        train_inputs.append(q_d_pair)
        train_labels.append(d[1])

train_inputs_tensor = torch.tensor(train_inputs)
train_labels_tensor = torch.tensor(train_labels)

if __name__ == "__main__":
    relevance_bert = BertForRelevance.from_pretrained(pretrained_weights)

    outputs = relevance_bert(train_inputs_tensor, labels=train_labels_tensor)
    # print(outputs)
    # print(pooled_output)
    # print(pooled_output.shape)
    # print(groud_truth_tensor)
    # print(groud_truth_tensor.shape)







