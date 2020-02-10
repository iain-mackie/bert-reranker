
from transformers import BertModel, BertForMaskedLM, BertConfig, BertTokenizer, PreTrainedEncoderDecoder
import torch
import os
# query

query = "I am a Test Query, please find the best passages (and entities!)"

# docs
good_doc = "I am a REALLY good document"
bad_doc = "I am a REALLY bad document"

# BERT init
pretrained_weights = 'bert-base-uncased'
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

query = "find me a good match for my query, please"
doc_good = "I'm a good match"
doc_bad = "I'm a bad match"



if __name__ == "__main__":
    # [CLS], Q, [SEP], D, [SEP]],
    test = "[CLS]" + query + "[SEP]" + doc_good + "[SEP]"
    print(test)

    test_encoded = tokenizer.encode(text=test, max_length=32, add_special_tokens=False)
    print(test_encoded)

    print(tokenizer.decode(test_encoded))