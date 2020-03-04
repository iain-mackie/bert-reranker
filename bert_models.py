

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

    if write:
        if os.path.isdir(exp_dir):
            print('*** Starting logging ***')
            exp_path = exp_dir + experiment_name + '/'
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
            total_loss += loss.sum().item()

            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Progress update every X batches.
            if step % logging_steps == 0 and not step == 0:
                log_epoch(t0=t0, step=step, total_steps=len(train_dataloader), loss_sum=total_loss,
                          device=device, labels=batch[3], scores=outputs[1])

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
                b_labels = batch[3].to(device, dtype=torch.float)

                with torch.no_grad():
                    outputs = model.forward(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                            attention_mask=b_attention_mask, labels=b_labels)
                loss = outputs[0]
                eval_loss += loss.sum().item()

                if device == torch.device("cpu"):
                    pred_list += flatten_list(outputs[1].cpu().detach().numpy().tolist())
                    label_list += batch[3].cpu().numpy().tolist()
                else:
                    pred_list += flatten_list(outputs[1].cpu().detach().numpy().tolist())
                    label_list += flatten_list(b_labels.cpu().numpy().tolist())

                # Progress update every X batches.
                if step % logging_steps == 0 and not step == 0:
                    log_epoch(t0=t0, step=step, total_steps=len(validation_dataloader), loss_sum=eval_loss,
                              device=device, labels=batch[3], scores=outputs[1])


            # Report the final accuracy for this validation run.
            avg_validation_loss = eval_loss / len(validation_dataloader)

            labels_groups, scores_groups, queries_groups, doc_ids_groups, rel_docs_groups = group_bert_outputs_by_query(
                score_list=pred_list, label_list=label_list, query_docids_map=query_docids_map, query_rel_doc_map=query_rel_doc_map)

            metrics_strings, label_metrics, bert_metrics, oracle_metrics = get_metrics(labels_groups=labels_groups,
                                                                                       scores_groups=scores_groups,
                                                                                       rel_docs_groups=rel_docs_groups)

            label_string = get_metrics_string(metrics_strings=metrics_strings, metrics=label_metrics, name='ORIGINAL')
            bert_string = get_metrics_string(metrics_strings=metrics_strings, metrics=bert_metrics, name='BERT')
            oracle_string = get_metrics_string(metrics_strings=metrics_strings, metrics=oracle_metrics, name='ORACLE')

            logging.info("")
            logging.info("  Average validation loss: {0:.5f}".format(avg_validation_loss))
            logging.info(label_string)
            logging.info(bert_string)
            logging.info(oracle_string)
            logging.info("  Validation took: {:}".format(format_time(time.time() - t0)))

            metrics.append('Epoch {} -  Average validation loss: '.format(str(epoch_i)) + str(avg_validation_loss) + '\n')
            metrics.append('Epoch {} -'.format(str(epoch_i)) + label_string + '\n')
            metrics.append('Epoch {} -'.format(str(epoch_i)) + bert_string + '\n')
            metrics.append('Epoch {} -'.format(str(epoch_i)) + oracle_string + '\n')


        else:

            logging.info('*** skipping validation ***')

        # Writing model & metrics
        if write:
            logging.info('Writing epoch model to file')
            if os.path.isdir(exp_dir):
                exp_path = exp_dir + experiment_name + '/'

                if os.path.isdir(exp_path) == False:
                    os.mkdir(exp_path)

                epoch_dir = exp_path + 'epoch{}/'.format(epoch_i)
                if os.path.isdir(epoch_dir) == False:
                    os.mkdir(epoch_dir)

                try:
                    model.module.save_pretrained(epoch_dir)
                except AttributeError:
                    model.save_pretrained(epoch_dir)

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


def run_metrics(validation_dataloader, run_path, qrels_path):

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

    # # static
    # batch_size = 16*3
    # pretrained_weights = 'bert-base-uncased'
    # relevance_bert = nn.DataParallel(BertReRanker.from_pretrained(pretrained_weights))
    # epochs = 15
    # eps = 1e-8
    # #lr = 5e-5
    # seed_val = 42
    # write = True
    # do_eval = True
    # logging_steps = 100
    # exp_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    # base_path = '/nfs/trec_car/data/bert_reranker_datasets/'
    # # run_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmarkY1.run'
    # # qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmarkY1.qrels'
    #
    # training_metadata = ['training_data/train_benchmarkY1_10_dataset_no_qrels_pad.pt',
    #                      'training_data/train_benchmarkY1_10_dataset_no_qrels.pt',
    #                      'training_data/train_benchmarkY1_10_dataset_with_qrels_pad.pt',
    #                      'training_data/train_benchmarkY1_10_dataset_with_qrels.pt']
    #
    # validation_metadata = [('dev_benchmarkY1.run', 'dev_benchmarkY1.qrels', 'dev_benchmarkY1.pt')]
    #                        # ('dev_benchmark_Y1_25.run', 'dev_benchmark_Y1_25.qrels', 'dev_benchmark_Y1_25_dataset.pt'),
    #                        # ('dev_benchmarkY1_100.run', 'dev_benchmarkY1_100.qrels', 'dev_benchmarkY1_100_dataset.pt')]

    # # loop over training & validation
    # for lr in [5e-5, 3e-5]:
    #     for r, q, pt in validation_metadata:
    #         for t in training_metadata:
    #
    #             train_file = t[14:]
    #             train_file = train_file[:len(train_file) - 3]
    #             experiment_name = 'exp_new_pipelines_V3_' + train_file + '_' + q[:len(q)-6] + '_' + str(lr)
    #
    #             dev_path = base_path + pt
    #             print('loading dev tensor: {}'.format(dev_path))
    #             validation_tensor = torch.load(dev_path)
    #             validation_dataloader = build_validation_data_loader(tensor=validation_tensor, batch_size=batch_size)
    #
    #             train_path = base_path + t
    #             print('loading train tensor: {}'.format(train_path))
    #             train_tensor = torch.load(train_path)
    #             train_dataloader = build_training_data_loader(tensor=train_tensor, batch_size=batch_size)
    #
    #             run_path = base_path + r
    #             qrels_path = base_path + q
    #
    #             fine_tuning_bert_re_ranker(model=relevance_bert, train_dataloader=train_dataloader,
    #                                        validation_dataloader=validation_dataloader, epochs=epochs, lr=lr, eps=eps,
    #                                        seed_val=seed_val, write=write, exp_dir=exp_dir, experiment_name=experiment_name,
    #                                        do_eval=do_eval, logging_steps=logging_steps, run_path=run_path,
    #                                        qrels_path=qrels_path)


    # test_path = '/nfs/trec_car/data/bert_reranker_datasets/test_100_dataset.pt'
    # print('loading test  tensor: {}'.format(test_path))
    # test_tensor = torch.load(test_path)
    # batch_size = 32*3
    # test_tensor = build_validation_data_loader(tensor=test_tensor, batch_size=batch_size)
    # run_path = '/nfs/trec_car/data/bert_reranker_datasets/test_100.run'
    # qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/test_100.qrels'
    # exp_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    # write_base = '/nfs/trec_car/data/bert_reranker_datasets/'
    # exp_metadata = [
    #     ('exp_new_pipelines_V3_train_benchmarkY1_10_dataset_no_qrels_dev_benchmarkY1_5e-05/epoch4/', 'test_100_new_pipeline_no_qrels.run'),
    #     ('exp_new_pipelines_V3_train_benchmarkY1_10_dataset_no_qrels_pad_dev_benchmarkY1_5e-05/epoch1/', 'test_100_new_pipeline_no_qrels_pad.run'),
    #     ('exp_new_pipelines_V3_train_benchmarkY1_10_dataset_with_qrels_dev_benchmarkY1_5e-05/epoch1/', 'test_100_new_pipeline_with_qrels.run'),
    #     ('exp_new_pipelines_V3_train_benchmarkY1_10_dataset_with_qrels_pad_dev_benchmarkY1_5e-05/epoch13/', 'test_100_new_pipeline_with_qrels_pad.run'),
    # ]
    # for m, w in exp_metadata:
    #     model_path = exp_path + m
    #     write_path = write_base + w
    #
    #     inference_bert_re_ranker(model_path=model_path, dataloader=test_tensor, run_path=run_path, qrels_path=qrels_path,
    #     write_path=write_path)

    test_path = '/nfs/trec_car/data/bert_reranker_datasets/test_dataset.pt'
    print('loading test  tensor: {}'.format(test_path))
    test_tensor = torch.load(test_path)
    batch_size = 64
    test_data_loader = build_validation_data_loader(tensor=test_tensor, batch_size=batch_size)
    run_path = '/nfs/trec_car/data/bert_reranker_datasets/test.run'
    qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/test.qrels'
    run_metrics(validation_dataloader=test_path, run_path=run_path, qrels_path=qrels_path)
