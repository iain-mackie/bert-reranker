

from transformers import BertModel, BertPreTrainedModel
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from pytorch_datasets import build_validation_data_loader, build_training_data_loader
from torch import nn, sigmoid
from torch.nn import MSELoss
from metrics import group_bert_outputs_by_query, get_metrics
from utils.logging_utils import get_metrics_string, log_epoch, format_time
from utils.trec_utils import write_trec_run,  get_query_docids_map, get_query_rel_doc_map, write_trec_eval
from utils.data_utils import flatten_list

import logging
import torch
import time
import numpy as np
import random
import os


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
                               seed_val=42, write=False, exp_dir=None, experiment_name='test', do_eval=True,
                               logging_steps=100, run_path=None, qrels_path=None):

    # Set the seed value all over the place to make this reproducible.
    print('starting fine tuning')
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    exp_path = exp_dir + experiment_name + '/'
    results_path = exp_path + 'results.txt'
    results_csv_path = exp_path + 'results.csv'

    if write:
        if os.path.isdir(exp_dir):
            print('*** Starting logging ***')
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

    logging.info('--- SETUP ---')
    setup_strings = ['epochs', 'lr', 'eps', 'seed_val', 'write', 'exp_dir', 'experiment_name', 'do_eval', 'logging_steps', 'run_path', 'qrels_path']
    setup_values = [epochs, lr, eps, seed_val, write, exp_dir, experiment_name, do_eval, logging_steps, run_path, qrels_path]
    for i in zip(setup_strings, setup_values):
        logging.info('{}: {}'.format(i[0], i[1]))
    logging.info('-------------')

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    query_docids_map = get_query_docids_map(run_path=run_path)
    query_rel_doc_map = get_query_rel_doc_map(qrels_path=qrels_path)

    for epoch_i in range(1, epochs+1):

        # ========================================
        #               Training
        # ========================================

        logging.info("=================================")
        logging.info('======== Epoch {:} / {:} ========'.format(epoch_i, epochs))
        logging.info("=================================")

        t0 = time.time()
        train_loss = 0
        model.train()

        for train_step, train_batch in enumerate(train_dataloader):

            b_input_ids = train_batch[0].to(device)
            b_token_type_ids = train_batch[1].to(device)
            b_attention_mask = train_batch[2].to(device)
            b_labels = train_batch[3].to(device, dtype=torch.float)

            model.zero_grad()
            outputs = model.forward(input_ids=b_input_ids, attention_mask=b_attention_mask,
                                    token_type_ids=b_token_type_ids, labels=b_labels)
            loss = outputs[0]
            train_loss += loss.sum().item()

            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Progress update every X batches.
            if ((train_step+1) % logging_steps == 0) or ((train_step+1) == len(train_dataloader)):

                metrics = []
                metrics_stats_headers = 'epoch,batch,'
                metrics_stats = [epoch_i, train_step+1]

                avg_train_loss = train_loss / len(train_dataloader)
                metrics.append('----- Epoch {} / Batch {} -----\n'.format(str(epoch_i), str(train_step+1)))
                metrics.append('Training loss: {}\n'.format(str(avg_train_loss)))

                metrics_stats_headers += 'train_loss,'
                metrics_stats.append(avg_train_loss)

                logging.info('----- Epoch {} / Batch {} -----\n'.format(str(epoch_i), str(train_step+1)))
                # log_epoch(t0=t0, step=train_step, total_steps=len(train_dataloader), loss_sum=train_loss,
                #           device=device, labels=train_batch[3], scores=outputs[1])
                logging.info("Training loss: {0:.5f}".format(avg_train_loss))
                logging.info("Training time: {:}".format(format_time(time.time() - t0)))

                if do_eval:

                    # ========================================
                    #               Validation
                    # ========================================

                    t0 = time.time()
                    dev_loss = 0

                    pred_list = []
                    label_list = []

                    model.eval()
                    for dev_step, dev_batch in enumerate(validation_dataloader):

                        b_input_ids = dev_batch[0].to(device)
                        b_token_type_ids = dev_batch[1].to(device)
                        b_attention_mask = dev_batch[2].to(device)
                        b_labels = dev_batch[3].to(device, dtype=torch.float)

                        with torch.no_grad():
                            outputs = model.forward(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                                    attention_mask=b_attention_mask, labels=b_labels)
                        loss = outputs[0]
                        dev_loss += loss.sum().item()

                        if device == torch.device("cpu"):
                            pred_list += flatten_list(outputs[1].cpu().detach().numpy().tolist())
                            label_list += dev_batch[3].cpu().numpy().tolist()
                        else:
                            pred_list += flatten_list(outputs[1].cpu().detach().numpy().tolist())
                            label_list += flatten_list(b_labels.cpu().numpy().tolist())

                    # Report the final accuracy for this validation run.
                    avg_validation_loss = dev_loss / len(validation_dataloader)

                    labels_groups, scores_groups, queries_groups, doc_ids_groups, rel_docs_groups = group_bert_outputs_by_query(
                        score_list=pred_list, label_list=label_list, query_docids_map=query_docids_map, query_rel_doc_map=query_rel_doc_map)

                    metrics_strings, label_metrics, bert_metrics, oracle_metrics = get_metrics(labels_groups=labels_groups,
                                                                                               scores_groups=scores_groups,
                                                                                               rel_docs_groups=rel_docs_groups)
                    metrics_stats_headers += 'dev_loss,'
                    metrics_stats.append(avg_validation_loss)

                    for l in [label_metrics, bert_metrics, oracle_metrics]:
                        for i in l:
                            metrics_stats.append(i)

                    for l in ['ORIGINAL', 'BERT', 'ORACLE']:
                        for s in metrics_strings:
                            metrics_stats_headers += l + '_' + s + ','

                    label_string = get_metrics_string(metrics_strings=metrics_strings, metrics=label_metrics, name='ORIGINAL')
                    bert_string = get_metrics_string(metrics_strings=metrics_strings, metrics=bert_metrics, name='BERT')
                    oracle_string = get_metrics_string(metrics_strings=metrics_strings, metrics=oracle_metrics, name='ORACLE')

                    logging.info("Validation loss: {0:.5f}".format(avg_validation_loss))
                    logging.info("Validation time: {:}".format(format_time(time.time() - t0)))
                    logging.info(label_string)
                    logging.info(bert_string)
                    logging.info(oracle_string)

                    metrics.append('Validation loss: ' + str(avg_validation_loss) + '\n')
                    metrics.append(' ' + label_string + '\n')
                    metrics.append(' ' + bert_string + '\n')
                    metrics.append(' ' + oracle_string + '\n')

                else:

                    logging.info('*** skipping validation ***')

                # Writing model & metrics
                if write:
                    logging.info('Writing epoch model to file')
                    if os.path.isdir(exp_dir):

                        if os.path.isdir(exp_path) == False:
                            os.mkdir(exp_path)

                        epoch_dir = exp_path + 'epoch{}_batch{}/'.format(epoch_i, train_step+1)
                        if os.path.isdir(epoch_dir) == False:
                            os.mkdir(epoch_dir)

                        try:
                            model.module.save_pretrained(epoch_dir)
                        except AttributeError:
                            model.save_pretrained(epoch_dir)

                        logging.info('writing epoch metrics')
                        f = open(results_path, "a+")
                        for m in metrics:
                            f.write(m)

                        if os.path.exists(results_csv_path) == False:
                            with open(results_csv_path, 'a+') as f:
                                f.write(metrics_stats_headers + '\n')

                        with open(results_csv_path, 'a+') as f:
                            for m in metrics_stats:
                                f.write('{0:.5f},'.format(m))
                            f.write('\n')

                    else:
                        logging.warning('MODEL PATH DOES NOT EXIST')
                else:
                    logging.warning('*** Not writing model to file ***')

    logging.info("")
    logging.info("Training complete!")


def inference_bert_re_ranker(model_path, dataloader, run_path, qrels_path, write_path):

    model = nn.DataParallel(BertReRanker.from_pretrained(model_path))
    query_docids_map = get_query_docids_map(run_path=run_path)
    query_rel_doc_map = get_query_rel_doc_map(qrels_path=qrels_path)

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
    label_list = []
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

        if device == torch.device("cpu"):
            pred_list += flatten_list(outputs.cpu().detach().numpy().tolist())
            label_list += batch[3].cpu().numpy().tolist()
        else:
            pred_list += flatten_list(outputs.cpu().detach().numpy().tolist())
            label_list += flatten_list(batch[3].cpu().numpy().tolist())

        if step % 100 == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}'.format(step, len(dataloader), elapsed))


    print('sorting groups')
    labels_groups, scores_groups, queries_groups, doc_ids_groups, rel_docs_groups = group_bert_outputs_by_query(
        score_list=pred_list, label_list=label_list, query_docids_map=query_docids_map, query_rel_doc_map=query_rel_doc_map)

    print('getting metrics')
    metrics_strings, label_metrics, bert_metrics, oracle_metrics = get_metrics(labels_groups=labels_groups,
                                                                               scores_groups=scores_groups,
                                                                               rel_docs_groups=rel_docs_groups)

    label_string = get_metrics_string(metrics_strings=metrics_strings, metrics=label_metrics, name='ORIGINAL')
    bert_string = get_metrics_string(metrics_strings=metrics_strings, metrics=bert_metrics, name='BERT')
    oracle_string = get_metrics_string(metrics_strings=metrics_strings, metrics=oracle_metrics, name='ORACLE')
    print(label_string)
    print(bert_string)
    print(oracle_string)

    print('writing groups')
    write_trec_run(scores_groups=scores_groups, queries_groups=queries_groups, doc_ids_groups=doc_ids_groups,
                   write_path=write_path)

    write_trec_eval(write_path, label_string, oracle_string, bert_string)


def run_metrics_from_dataloader(validation_dataloader, run_path, qrels_path):

    query_docids_map = get_query_docids_map(run_path=run_path)
    query_rel_doc_map = get_query_rel_doc_map(qrels_path=qrels_path)

    label_list = []
    for step, batch in enumerate(validation_dataloader):
        label_list += flatten_list(batch[3].cpu().numpy().tolist())

    labels_groups, scores_groups, queries_groups, doc_ids_groups, rel_docs_groups = group_bert_outputs_by_query(
        score_list=label_list, label_list=label_list, query_docids_map=query_docids_map,
        query_rel_doc_map=query_rel_doc_map)

    metrics_strings, label_metrics, _, oracle_metrics = get_metrics(labels_groups=labels_groups,
                                                                    scores_groups=scores_groups,
                                                                    rel_docs_groups=rel_docs_groups)

    label_string = get_metrics_string(metrics_strings=metrics_strings, metrics=label_metrics, name='RANKING')
    oracle_string = get_metrics_string(metrics_strings=metrics_strings, metrics=oracle_metrics, name='ORACLE')

    print(label_string)
    print(oracle_string)

    return label_string




if __name__ == "__main__":
    # run_path = os.path.join(os.getcwd(), 'test_data', 'test_model.run')
    # qrels_path = os.path.join(os.getcwd(), 'test_data', 'test_model.qrels')
    # run_metrics(run_path, qrels_path)

    #static
    # batch_size = 8*3
    # pretrained_weights = 'bert-base-uncased'
    # relevance_bert = nn.DataParallel(BertReRanker.from_pretrained(pretrained_weights))
    # epochs = 4
    # eps = 1e-8
    # lr_list = [1e-5]
    # seed_val = 42
    # write = True
    # do_eval = True
    # logging_steps = 10000
    # exp_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    #
    # for i in ['1000', '2500', '5000', '10000']:
    #     for lr in lr_list:
    #
    #         train_path = '/nfs/trec_car/data/bert_reranker_datasets/training_data_sample_queries/train_fold_0_train_hierarchical_{}_random_queries_dataset.pt'.format(i)
    #         dev_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmarkY1.pt'
    #         run_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmarkY1.run'
    #         qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmarkY1.qrels'
    #         experiment_name = 'random_queries_{}_dev_10'.format(i) + '_' + str(lr)
    #
    #         print('loading dev tensor: {}'.format(dev_path))
    #         validation_tensor = torch.load(dev_path)
    #         validation_dataloader = build_validation_data_loader(tensor=validation_tensor, batch_size=batch_size)
    #
    #         print('loading train tensor: {}'.format(train_path))
    #         train_tensor = torch.load(train_path)
    #         train_dataloader = build_training_data_loader(tensor=train_tensor, batch_size=batch_size)
    #
    #         fine_tuning_bert_re_ranker(model=relevance_bert, train_dataloader=train_dataloader,
    #                                     validation_dataloader=validation_dataloader, epochs=epochs, lr=lr, eps=eps,
    #                                     seed_val=seed_val, write=write, exp_dir=exp_dir, experiment_name=experiment_name,
    #                                     do_eval=do_eval, logging_steps=logging_steps, run_path=run_path,
    #                                     qrels_path=qrels_path)

    exp_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    write_base = '/nfs/trec_car/data/bert_reranker_datasets/test_runs/'
    exp_metadata = [
        ('random_queries_500_dev_10_1e-05/epoch1_batch4013/', '500'),
        ('random_queries_1000_dev_10_1e-05/epoch3_batch8109/', '1000'),
    ]
    for t in [10, 100, 1000]:
        for m, desc in exp_metadata:
            if t == 1000:
                test_path = '/nfs/trec_car/data/bert_reranker_datasets/test_dataset.pt'
                run_path = '/nfs/trec_car/data/bert_reranker_datasets/test.run'
                qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/test.qrels'
            else:
                test_path = '/nfs/trec_car/data/bert_reranker_datasets/test_{}_dataset.pt'.format(t)
                run_path = '/nfs/trec_car/data/bert_reranker_datasets/test_{}.run'.format(t)
                qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/test_{}.qrels'.format(t)

            print('loading test  tensor: {}'.format(test_path))
            test_tensor = torch.load(test_path)
            batch_size = 32 * 3
            test_tensor = build_validation_data_loader(tensor=test_tensor, batch_size=batch_size)

            model_path = exp_path + m
            write_path = write_base + 'test_random_queries_test_{}_train_{}'.format(t, desc)

            inference_bert_re_ranker(model_path=model_path, dataloader=test_tensor, run_path=run_path, qrels_path=qrels_path,
                                     write_path=write_path)

    # test_path = '/nfs/trec_car/data/bert_reranker_datasets/test_dataset.pt'
    # print('loading test  tensor: {}'.format(test_path))
    # test_tensor = torch.load(test_path)
    # batch_size = 64
    # test_dataloader = build_validation_data_loader(tensor=test_tensor, batch_size=batch_size)
    # run_path = '/nfs/trec_car/data/bert_reranker_datasets/test.run'
    # qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/test.qrels'
    # run_metrics(validation_dataloader=test_dataloader, run_path=run_path, qrels_path=qrels_path)
