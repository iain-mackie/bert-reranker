
from transformers import BertModel, BertPreTrainedModel
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from preprocessing import build_data_loader
from torch import nn, sigmoid
from torch.nn import MSELoss

import logging
import torch
import time
import datetime
import numpy as np
import random
import os
import itertools
import collections


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

    def pred(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
             inputs_embeds=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        logits = sigmoid(self.relevance_pred(pooled_output))

        return logits




def format_time(elapsed):

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def fine_tuning_bert_re_ranker(model, train_dataloader, validation_dataloader, epochs=5, lr=5e-5, eps=1e-8,
                               seed_val=42, write=False, model_path=None, experiment_name='test', do_eval=False,
                               logging_steps=100):
    # Set the seed value all over the place to make this reproducible.
    print('starting fine tuning')
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if write:
        if os.path.isdir(model_path):
            print('*** Staring logging ***')
            exp_path = model_path + experiment_name + '/'
            if os.path.isdir(exp_path) == False:
                os.mkdir(exp_path)
            logging_path = exp_path + 'output.log'
            logging.basicConfig(filename=logging_path, level=logging.DEBUG)

    # If there's a GPU available...
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        logging.info('There are %d GPU(s) available.' % torch.cuda.device_count())
        logging.info('We will use the GPU: {}'.format(torch.cuda.get_device_name(0)))

        model.cuda()

    # If not...
    else:

        logging.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_values = []

    for epoch_i in range(0, epochs):

        metrics = []

        # ========================================
        #               Training
        # ========================================

        logging.info("")
        logging.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logging.info('Training...')

        t0 = time.time()

        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_token_type_ids = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_labels = batch[3].to(device, dtype=torch.float)

            model.zero_grad()

            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, token_type_ids=b_token_type_ids,
                            labels=b_labels)

            loss = outputs[0]

            total_loss += loss.item()

            # Progress update every 250 batches.
            if step % logging_steps == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    MSE:  {}'.format(
                    step, len(train_dataloader), elapsed, total_loss/step+1))
                logging.info('      Prediction : {} '.format(outputs[1].cpu().detach().numpy().tolist()))
                logging.info('      Labels     : {} '.format(b_labels.cpu().numpy().tolist()))


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        metrics.append('Average training loss: ' + str(avg_train_loss) + '\n')

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        logging.info("")
        logging.info("  Average training loss: {0:.5f}".format(avg_train_loss))
        logging.info("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        if do_eval:

            logging.info("---")
            logging.info('Validation...')

            t0 = time.time()

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0

            for batch in validation_dataloader:

                b_input_ids = batch[0].to(device)
                b_token_type_ids = batch[1].to(device)
                b_attention_mask = batch[2].to(device)
                b_labels = batch[3].to(device, dtype=torch.float)

                with torch.no_grad():
                    outputs = model(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask,
                                    labels=b_labels)

                loss = outputs[0]
                eval_loss += loss

                # Track the number of batches
                nb_eval_steps += 1

            # Report the final accuracy for this validation run.
            avg_validation_loss = eval_loss / nb_eval_steps
            logging.info("")
            logging.info("  Average validation loss: {0:.5f}".format(avg_validation_loss))
            logging.info("  Validation took: {:}".format(format_time(time.time() - t0)))

            metrics.append('Average validation loss: ' + str(avg_validation_loss) + '\n')


        else:
            logging.info('*** skipping validation ***')

        # Writing model & metrics
        if write:
            logging.info('Writing epoch model to file')
            if os.path.isdir(model_path):
                exp_path = model_path + experiment_name + '/'

                if os.path.isdir(exp_path) == False:
                    os.mkdir(exp_path)

                epoch_dir = exp_path + 'epoch{}/'.format(epoch_i)
                if os.path.isdir(epoch_dir) == False:
                    os.mkdir(epoch_dir)

                model.save_pretrained(epoch_dir)  # save model

                logging.info('writing epoch metrics')
                results_path = epoch_dir + 'results.txt'
                f = open(results_path, "a+")
                for m in metrics:
                    f.write(m)
                f.close()

            else:
                logging.warning('MODEL PATH DOES NOT EXIST')
        else:
            logging.warning('*** Not writing model to file ***')

    logging.info("")
    logging.info("Training complete!")

    # TODO - trec output wrtiter


def flatten_list(l):
    return list(itertools.chain(*l))


def inference_bert_re_ranker(model_path, dataloader, query_docids_map, run_path, num_rank=10):

    model = BertReRanker.from_pretrained(model_path)

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

    pred_list = []
    counter_written = 0

    model.eval()

    #total_steps = len(dataloader)

    run_file = open(run_path, 'a+')
    for step, batch in enumerate(dataloader):

        b_input_ids = batch[0].to(device)
        b_token_type_ids = batch[1].to(device)
        b_attention_mask = batch[2].to(device)

        with torch.no_grad():
            outputs = model.pred(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                 attention_mask=b_attention_mask)

        pred_list += flatten_list(outputs.cpu().detach().numpy().tolist())


        possible_write = len(pred_list) // num_rank

        while counter_written < possible_write:

            start_idx = counter_written * num_rank
            end_idx = counter_written * num_rank + num_rank

            scores = pred_list[start_idx:end_idx]
            query_docids = query_docids_map[start_idx:end_idx]

            queries, doc_ids = zip(*query_docids)
            assert len(set(queries)) == 1, "Queries must be all the same."
            query = queries[0]

            d = {i[0]:i[1] for i in zip(doc_ids, scores)}
            od = collections.OrderedDict(sorted(d.items(), key=lambda item: item[1], reverse=True))

            rank = 1
            for doc_id in od.keys():

                output_line = " ".join((query, "Q0", str(doc_id), str(rank), str(od[doc_id]), "BERT"))
                run_file.write(output_line + "\n")
                rank += 1

            counter_written += 1

    run_file.close()


def get_query_docids_map(set_name, data_path):

    run_path = os.path.join(data_path, set_name + ".run")

    query_docids_map = []
    with open(run_path) as ref_file:

        for line in ref_file:
            query, _, doc_id, _, _, _ = line.strip().split(" ")

            query_docids_map.append((query, doc_id))

    return query_docids_map


def trec_output():
    pass


if __name__ == "__main__":

    train_path = os.path.join(os.getcwd(), 'toy_dev_dataset.pt')
    dev_path = os.path.join(os.getcwd(), 'toy_dev_dataset.pt')

    train_tensor = torch.load(train_path)
    validation_tensor = torch.load(dev_path)
    batch_size = 8

    train_dataloader, validation_dataloader = build_data_loader(train_tensor=train_tensor,
                                                                validation_tensor=validation_tensor,
                                                                batch_size=batch_size)
    #
    #
    #
    # pretrained_weights = 'bert-base-uncased'
    # relevance_bert = BertReRanker.from_pretrained(pretrained_weights)
    # epochs = 5
    # lr = 5e-5
    # eps = 1e-8
    # seed_val = 42
    # write = True
    # model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    # experiment_name = 'toy_bert_run'
    # fine_tuning_bert_re_ranker(model=relevance_bert, train_dataloader=train_dataloader,
    #                            validation_dataloader=validation_dataloader, epochs=epochs, lr=lr, eps=eps,
    #                            seed_val=seed_val, write=write, model_path=model_path, experiment_name=experiment_name)

    set_name = 'toy_train'
    data_path = '/Users/iain/LocalStorage/coding/github/bert-reranker'
    query_docids_map = get_query_docids_map(set_name, data_path)

    print(query_docids_map)
    print(len(query_docids_map))
    model_path = os.path.join(os.getcwd(), 'models', 'test_preds_4', 'epoch1')
    run_path = os.path.join(os.getcwd(), 'bert.run')
    inference_bert_re_ranker(model_path=model_path, dataloader=validation_dataloader, query_docids_map=query_docids_map,
                             run_path=run_path, num_rank=10)


