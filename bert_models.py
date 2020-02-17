
from transformers import BertModel, BertPreTrainedModel
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from preprocessing import build_data_loader
from torch import nn, sigmoid
from torch.nn import MSELoss

import torch
import time
import datetime
import numpy as np
import random
import os

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


def format_time(elapsed):

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def fine_tuning_bert_re_ranker(model, train_dataloader, validation_dataloader, epochs=5, lr=5e-5, eps=1e-8,
                               seed_val=42, write=False, model_path=None, experiment_name='test'):
    # Set the seed value all over the place to make this reproducible.

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # If there's a GPU available...
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

        model.cuda()

    # If not...
    else:

        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_values = []

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_token_type_ids = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_labels = batch[3].to(device, dtype=torch.float)

            model.zero_grad()

            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, token_type_ids=b_token_type_ids,
                            labels=b_labels)

            loss = outputs[0]

            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print(avg_train_loss)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================

        t0 = time.time()

        model.eval()
        #eval_loss = 0
        #eval_accuracy = 0
        #nb_eval_steps = 0
        #nb_eval_examples = 0, 0

        for batch in validation_dataloader:

            b_input_ids = batch[0].to(device)
            b_token_type_ids = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_labels = batch[3].to(device, dtype=torch.float)

            with torch.no_grad():
                outputs = model(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask,
                                labels=b_labels)

            loss = outputs[0]
            print('*** LOSS ***')
            print(loss)

            print('*** PRED ***')
            pred = outputs[1].numpy().tolist()
            print(pred)

            print('*** GT ***')
            gt = b_labels.numpy().tolist()
            print(gt)

            # Calculate the accuracy for this batch of test sentences.
            # tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            # eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            #nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        # print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

        # Writing model
        if write:
            print('Writing model to file')
            if os.path.isdir(model_path):
                exp_path = os.path.join(model_path, experiment_name)

                if os.path.isdir(exp_path) == False:
                    os.mkdir(exp_path)

                epoch_dir = os.path.join(exp_path, 'epoch{}'.format(epoch_i))
                os.mkdir(epoch_dir)
                model.save_pretrained(epoch_dir)  # save
            else:
                print('MODEL PATH DOES NOT EXIST')
        else:
            print('Not writing model to file')

    print("")
    print("Training complete!")

    # TODO - trec output wrtiter

def inference_bert_re_ranker(model_path):

    model = BertReRanker.from_pretrained(model_path)

    print(model)

def trec_output():
    pass


if __name__ == "__main__":

    train_path = '/nfs/trec_car/data/bert_reranker_datasets/test_dataset_from_pickle.pt'
    dev_path = '/nfs/trec_car/data/bert_reranker_datasets/test_dataset_from_pickle.pt'

    train_tensor = torch.load(train_path)
    validation_tensor = torch.load(dev_path)

    train_dataloader, validation_dataloader = build_data_loader(train_tensor=train_tensor,
                                                                validation_tensor=validation_tensor,
                                                                batch_size=8)

    pretrained_weights = 'bert-base-uncased'
    relevance_bert = BertReRanker.from_pretrained(pretrained_weights)

    fine_tuning_bert_re_ranker(model=relevance_bert, train_dataloader=train_dataloader,
                               validation_dataloader=validation_dataloader, epochs=5, lr=5e-5, eps=1e-8, seed_val=42,
                               write=False, model_path=None, experiment_name='test')