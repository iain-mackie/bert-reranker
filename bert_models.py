
from transformers import BertModel, BertPreTrainedModel

from torch import nn
from torch import sigmoid
from torch.nn import MSELoss

#TODO - cosine similarity of q & d

#TODO - sentence relevance aggregation (See Birch implmenentation) --> BERT sentence level OR a(BM25+RM3) + (1-a)(S1+...)

#TODO - add comments and change docstring


class BertReRanker(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.relevance_pred = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = sigmoid(self.relevance_pred(pooled_output))

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

