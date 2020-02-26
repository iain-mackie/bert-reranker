

from transformers import BertModel, BertPreTrainedModel
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from bert_utils import build_validation_data_loader, build_training_data_loader
from torch import nn, sigmoid
from torch.nn import MSELoss
from metrics import group_bert_outputs_by_query, get_metrics, write_trec_run
from bert_utils import format_time, flatten_list, get_query_docids_map
import logging
import torch
import time
import numpy as np
import random
import os
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



def fine_tuning_bert_re_ranker(model, train_dataloader, validation_dataloader, epochs=5, lr=5e-5, eps=1e-8,
                               seed_val=42, write=False, model_dir=None, experiment_name='test', do_eval=True,
                               logging_steps=100, run_path=None):
    # Set the seed value all over the place to make this reproducible.
    print('starting fine tuning')
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if write:
        if os.path.isdir(model_dir):
            print('*** Staring logging ***')
            exp_path = model_dir + experiment_name + '/'
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

    query_docids_map = get_query_docids_map(run_path=run_path)

    loss_values = []
    epoch_i = 0
    for i in range(0, epochs):

        epoch_i += 1

        metrics = []
        metrics.append('------------------------\n')

        # ========================================
        #               Training
        # ========================================

        logging.info("")
        logging.info('======== Epoch {:} / {:} ========'.format(epoch_i, epochs))
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
            outputs = model.forward(input_ids=b_input_ids, attention_mask=b_attention_mask,
                                    token_type_ids=b_token_type_ids, labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()

            # Progress update every 250 batches.
            if step % logging_steps == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    MSE:  {}'.format(
                    step, len(train_dataloader), elapsed, total_loss/(step+1)))
                logging.info('      Prediction : {} '.format(outputs[1].cpu().detach().numpy().tolist()))
                logging.info('      Labels     : {} '.format(b_labels.cpu().numpy().tolist()))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        metrics.append('Epoch {} -  Average training loss: '.format(str(epoch_i)) + str(avg_train_loss) + '\n')
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
            eval_loss = 0

            pred_list = []
            label_list = []

            model.eval()
            for step, batch in enumerate(validation_dataloader):

                b_input_ids = batch[0].to(device)
                b_token_type_ids = batch[1].to(device)
                b_attention_mask = batch[2].to(device)

                with torch.no_grad():
                    outputs = model.pred(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                         attention_mask=b_attention_mask)
                loss = outputs[0]
                eval_loss += loss.item()

                if step % logging_steps == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    MSE:  {}'.format(
                        step, len(validation_dataloader), elapsed, eval_loss/(step+1)))

                if device == torch.device("cpu"):
                    pred_list += flatten_list(outputs.cpu().detach().numpy().tolist())
                    label_list += batch[3].cpu().numpy().tolist()
                else:
                    pred_list += flatten_list(outputs.cpu().detach().numpy().tolist())
                    label_list += flatten_list(batch[3].cpu().numpy().tolist())

            # Report the final accuracy for this validation run.
            avg_validation_loss = eval_loss / len(validation_dataloader)

            labels_groups, scores_groups, queries_groups, doc_ids_groups = group_bert_outputs_by_query(
                score_list=pred_list, label_list=label_list, query_docids_map=query_docids_map)

            string_labels, label_metrics, bert_metrics = get_metrics(labels_groups=labels_groups,
                                                                     scores_groups=scores_groups)

            def get_metrics_string(string_labels, metrics, name='BERT'):
                s = '  Average {}:  '.format(name)
                for i in zip(string_labels, metrics):
                    s += i[0] + ': {0:.5f}, '.format(i[1])
                return s

            label_string = get_metrics_string(string_labels=string_labels, metrics=label_metrics, name='LABELS')
            bert_string = get_metrics_string(string_labels=string_labels, metrics=bert_metrics, name='BERT')

            logging.info("")
            logging.info("  Average validation loss: {0:.5f}".format(avg_validation_loss))
            logging.info(label_string)
            logging.info(bert_string)
            logging.info("  Validation took: {:}".format(format_time(time.time() - t0)))

            metrics.append('Epoch {} -  Average validation loss: '.format(str(epoch_i)) + str(avg_validation_loss) + '\n')
            metrics.append('Epoch {} -'.format(str(epoch_i)) + label_string + '\n')
            metrics.append('Epoch {} -'.format(str(epoch_i)) + bert_string + '\n')

        else:

            logging.info('*** skipping validation ***')

        # Writing model & metrics
        if write:
            logging.info('Writing epoch model to file')
            if os.path.isdir(model_dir):
                exp_path = model_dir + experiment_name + '/'

                if os.path.isdir(exp_path) == False:
                    os.mkdir(exp_path)

                epoch_dir = exp_path + 'epoch{}/'.format(epoch_i)
                if os.path.isdir(epoch_dir) == False:
                    os.mkdir(epoch_dir)

                model.save_pretrained(epoch_dir)  # save model

                logging.info('writing epoch metrics')
                results_path = exp_path + 'results.txt'
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


def inference_bert_re_ranker(model_path, dataloader, run_path, write_path):

    model = nn.DataParallel(BertReRanker.from_pretrained(model_path))
    query_docids_map = get_query_docids_map(run_path=run_path)

    # If there's a GPU available...
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

        model.cuda()
        #model.to(device)

    # If not...
    else:

        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    pred_list = []

    model.eval()

    print('beginining inference')
    t0 = time.time()
    for step, batch in enumerate(dataloader):

        b_input_ids = batch[0].to(device)
        b_token_type_ids = batch[1].to(device)
        b_attention_mask = batch[2].to(device)
        b_labels = batch[3].to(device, dtype=torch.float)

        with torch.no_grad():
            # outputs = model.pred(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
            #                      attention_mask=b_attention_mask)
            _, outputs = model.forward(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                       attention_mask=b_attention_mask, labels=b_labels)

        pred_list += flatten_list(outputs.cpu().detach().numpy().tolist())

        if step % 1000 == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}'.format(step, len(dataloader), elapsed))

    fake_lables = [0] * len(pred_list)

    print('sorting groups')
    labels_groups, scores_groups, queries_groups, doc_ids_groups = group_bert_outputs_by_query(
        score_list=pred_list, label_list=fake_lables, query_docids_map=query_docids_map)

    print('writing groups')
    write_trec_run(scores_groups=scores_groups, queries_groups=queries_groups, doc_ids_groups=doc_ids_groups,
                   write_path=write_path)


if __name__ == "__main__":

    # train_path = '/nfs/trec_car/data/bert_reranker_datasets/train_benchmarkY1_0.5.pt'
    dev_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmarkY1.pt'

    # print('loading train tensor: {}'.format(train_path))
    # train_tensor = torch.load(train_path)
    print('loading dev tensor: {}'.format(dev_path))
    validation_tensor = torch.load(dev_path)
    batch_size = 16

    # train_dataloader = build_training_data_loader(tensor=train_tensor, batch_size=batch_size)
    validation_dataloader = build_validation_data_loader(tensor=validation_tensor, batch_size=batch_size)

    # print('running training & validation')
    # pretrained_weights = 'bert-base-uncased'
    # relevance_bert = BertReRanker.from_pretrained(pretrained_weights)
    # epochs = 5
    # lr = 1e-5
    # eps = 1e-8
    # seed_val = 42
    # write = True
    # model_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    # experiment_name = 'benchmarkY1_5'
    # do_eval = True
    # logging_steps = 100
    # run_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmarkY1.run'
    # fine_tuning_bert_re_ranker(model=relevance_bert, train_dataloader=train_dataloader, validation_dataloader=validation_dataloader,
    #                            epochs=epochs, lr=lr, eps=eps, seed_val=seed_val, write=write, model_dir=model_dir,
    #                            experiment_name=experiment_name, do_eval=do_eval, logging_steps=logging_steps, run_path=run_path)

    # test_path = '/nfs/trec_car/data/bert_reranker_datasets/test_dataset.pt'
    # print('loading test  tensor: {}'.format(test_path))
    # test_tensor = torch.load(test_path)
    # batch_size = 8
    # test_tensor = build_validation_data_loader(tensor=test_tensor, batch_size=batch_size)
    model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/benchmarkY1_5/epoch4/'
    write_path =  '/nfs/trec_car/data/bert_reranker_datasets/exp/benchmarkY1_5/bert_epoch4_dev_multi.run'
    run_path = '/nfs/trec_car/data/bert_reranker_datasets/test.run'
    inference_bert_re_ranker(model_path=model_path, dataloader=validation_dataloader, run_path=run_path, write_path=write_path)


