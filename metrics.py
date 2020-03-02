
import numpy as np


def get_map(run, R):

    correct_docs_sum = sum(run)
    if (correct_docs_sum > 0.0) and (R > 0):
        precision_sum = 0
        for i, r in enumerate(run):
            if r == 1.0:
                precision_sum += get_precision(run=run, k=(i+1))
        return precision_sum / R
    else:
        return 0.0


def get_R_prec(run, R):

    if R > 0:
        r_run = run[:R]
        return sum(r_run) / R
    else:
        return 0.0


def get_recip_rank(run):

    for i, r in enumerate(run):
        if r == 1.0:
            return 1/(i+1)
    return 0.0


def get_precision(run, k=20):

    k_run = run[:k]
    return sum(k_run) / k


def get_recall(run, R, k=40):

    k_run = run[:k]
    correct_docs_sum = sum(k_run)
    if R > 0:
        return correct_docs_sum / R
    return 1.0


def get_ndcg(run, R, k=20):

    k_run = run[:k]
    i_dcg, dcg = 0, 0
    num_rel = sum(run)
    if (num_rel > 0) and (R > 0):
        for i, r in enumerate(k_run):
            if i == 0:
                if (i+1) <= R:
                    i_dcg += 1
                dcg += r
            else:
                discount = np.log2(i+2)
                if (i+1) <= R:
                    i_dcg += 1 / discount
                dcg += r / discount
        return dcg / i_dcg
    else:
        return 0


def get_bert_labels(labels, scores):

    bert_labels = []
    ordered_scores = sorted(list(set(scores)), reverse=True)
    for os in ordered_scores:
        ixs = [i for i, x in enumerate(scores) if x == os]
        for i in ixs:
            bert_labels.append(labels[i])
    return bert_labels


def get_query_metrics(metrics, rank, R):

    metrics[0] += get_map(run=rank, R=R)
    metrics[1] += get_R_prec(run=rank, R=R)
    metrics[2] += get_recip_rank(run=rank)
    metrics[3] += get_precision(run=rank, k=20)
    metrics[4] += get_recall(run=rank, k=40, R=R)
    metrics[5] += get_ndcg(run=rank, R=R, k=20)

    return metrics


def get_metrics(labels_groups, scores_groups, rel_docs_groups):

    original_metrics = [0,0,0,0,0,0]
    bert_metrics = [0,0,0,0,0,0]
    oracle_metrics = [0,0,0,0,0,0]
    num_queries = len(labels_groups)
    metric_strings = ['map', 'R_prec', 'recip_rank', 'precision_20', 'recall_40', 'ndcg_20']

    for original_rank, scores, R_list in zip(labels_groups, scores_groups, rel_docs_groups):
        R = R_list[0]
        bert_rank = get_bert_labels(labels=original_rank, scores=scores)
        oracle_rank = sorted(original_rank, reverse=True)

        original_metrics = get_query_metrics(metrics=original_metrics, rank=original_rank, R=R)
        bert_metrics = get_query_metrics(metrics=bert_metrics, rank=bert_rank, R=R)
        oracle_metrics = get_query_metrics(metrics=oracle_metrics, rank=oracle_rank, R=R)

    def average_metrics(metrics, total):
        return [i / total for i in metrics]

    original_metrics = average_metrics(metrics=original_metrics, total=num_queries)
    bert_metrics = average_metrics(metrics=bert_metrics, total=num_queries)
    oracle_metrics = average_metrics(metrics=oracle_metrics, total=num_queries)

    return metric_strings, original_metrics, bert_metrics, oracle_metrics


def group_bert_outputs_by_query(label_list, score_list, query_docids_map, query_rel_doc_map):

    last_query = 'Not a query'
    labels_groups, scores_groups, queries_groups, doc_ids_groups, rel_docs_groups = [], [], [], [], []
    labels, scores, queries, doc_ids = [], [], [], []
    doc_counter = 0
    for i in zip(label_list, score_list, query_docids_map):
        query = i[2][0]

        if (doc_counter > 0) and (last_query != query):

            doc_counter = 0
            labels_groups.append(labels)
            scores_groups.append(scores)
            queries_groups.append(queries)
            doc_ids_groups.append(doc_ids)
            if last_query in query_rel_doc_map:
                sum_rel_docs = len(query_rel_doc_map[last_query])
                rel_docs_groups.append([sum_rel_docs])
            else:
                rel_docs_groups.append([0])
            labels, scores, queries, doc_ids = [], [], [], []

        labels.append(i[0])
        scores.append(i[1])
        queries.append(query)
        doc_ids.append(i[2][1])

        last_query = query
        doc_counter += 1

    labels_groups.append(labels)
    scores_groups.append(scores)
    queries_groups.append(queries)
    doc_ids_groups.append(doc_ids)
    if last_query in query_rel_doc_map:
        sum_rel_docs = len(query_rel_doc_map[last_query])
        rel_docs_groups.append([sum_rel_docs])
    else:
        rel_docs_groups.append([0])

    return labels_groups, scores_groups, queries_groups, doc_ids_groups, rel_docs_groups


if __name__ == '__main__':
    from utils.logging_utils import get_results_string

    l1 = [1, 2, 3, 4, 5]
    l2 = [0.121212121, 0.34334343434, 0.5656565656, 0.67676767, 0.8989898989]
    print(get_results_string(labels=l1, scores=l2))
    # run1 = [1,0,0]
    # run2 = [1, 0, 1, 0, 1]
    # run3 = [0, 0, 0, 0]
    # run4 = [0, 0, 1, 0, 1]
    # run5 = [1, 1, 0, 1, 0]
    #
    #
    # for r in [run1, run2, run3, run4, run5]:
    #     print('--------------------------')
    #     k = 4
    #     R = 4
    #     print(r)
    #     map = get_map(r, R=R)
    #     print('map: {}'.format(map))
    #     R_prec = get_R_prec(r, R=R)
    #     print('R_prec: {}'.format(R_prec))
    #     recip_rank = get_recip_rank(r)
    #     print('recip_rank: {}'.format(recip_rank))
    #     precision = get_precision(r, k=k)
    #     print('precision@{}: {}'.format(k, precision))
    #     ndcg = get_ndcg(r, k=k, R=R)
    #     print('ndcg@{}: {}'.format(k, ndcg))
    #     recall = get_recall(r, R=R, k=k)
    #     print('recall@{}: {}'.format(k, recall))
    #
    #
    #
    #
    #
