
from pytorch_datasets import flatten_list

import datetime
import logging
import torch
import time


def get_metrics_string(metrics_strings, metrics, name='BERT'):

    s = '  {}:  '.format(name)
    for i in zip(metrics_strings, metrics):
        s += i[0] + ': {0:.4f}, '.format(i[1])
    return s


def get_results_string(labels, scores):

    s = '  '
    for i in zip(labels, scores):
        s += '(truth: {:.3f}, pred: {:.3f}), '.format(i[0], i[1])
    return s


def format_time(elapsed):
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def log_epoch(t0, step, total_steps, loss_sum, device, labels, scores):
    elapsed = format_time(time.time() - t0)
    av_loss = loss_sum / (step+1)
    logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    MSE:  {}'.format(
        step, total_steps, elapsed, av_loss))

    if device == torch.device("cpu"):
        logging.info(get_results_string(labels=labels.cpu().numpy().tolist(),
                                        scores=flatten_list(scores.cpu().detach().numpy().tolist())))
    else:
        logging.info(get_results_string(labels=flatten_list(labels.cpu().numpy().tolist()),
                                        scores=flatten_list(scores.cpu().detach().numpy().tolist())))