
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import torch
from fine_tunning import build_data_loader, train_bert_relevance_model
from bert_models import BertReRanker

# BERT init
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

# Fake data
queries = ["find me a good match for my query, please"]
docs = [("Great query I am", 1.0),
        ("Yup, I'm a rubbish query", 0.0),
        ("Okay query for sure", 1.0),
        ("Who knows", 0.0)
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

    # Construct data loaders
    both_tensor = TensorDataset(input_ids_tensor, token_type_ids_tensor, attention_mask_tensor, labels_tensor)
    train_dataloader, validation_dataloader = build_data_loader(train_tensor=both_tensor, validation_tensor=both_tensor,
                                                                batch_size=2)

    # Init Bert Re-Ranker
    relevance_bert = BertReRanker.from_pretrained(pretrained_weights)

    # Train & validation run
    train_bert_relevance_model(model=relevance_bert, train_dataloader=train_dataloader,
                               validation_dataloader=validation_dataloader, epochs=2, lr=5e-4, eps=1e-8)

