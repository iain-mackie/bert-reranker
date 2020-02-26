
import collections
import numpy as np

#TODO - test metrics

def get_map(run):

    correct_docs_sum = sum(run)
    if correct_docs_sum > 0.0:
        rank = 1
        correct_docs = 1
        correct_docs_weighted_sum = 0
        for r in run:
            assert r == 0.0 or r == 1.0, 'score not 1.0 or 0.0'
            if r == 1.0:
                correct_docs_weighted_sum += correct_docs / rank
                correct_docs += 1
            rank += 1
        return correct_docs_weighted_sum * (1 / correct_docs_sum)
    else:
        return 0.0


def get_R_prec(run):
    R = sum(run)
    if R > 0:
        r_run = run[:int(R)]
        return sum(r_run) / R
    else:
        return 0.0


def get_recip_rank(run):
    rank = 1
    for r in run:
        if r == 1.0:
            return 1/rank
        rank += 1
    return 0.0


def get_precision(run, k=20):
    k_run = run[:k]
    return sum(k_run) / k


def get_recall(run, k=40):
    # TODO - need qrels
    k_run = run[:k]
    return None


def get_ndcg(run, k=20):
    k_run = run[:k]
    i_dcg = 0
    dcg = 0
    rank = 1
    for r in k_run:
        i_dcg += 1 / np.log2(rank + 1)
        dcg += r / np.log2(rank + 1)
        rank += 1
    return dcg / i_dcg


def get_bert_labels(labels, scores):
    bert_labels = []
    ordered_scores = sorted(list(set(scores)), reverse=True)
    for os in ordered_scores:
        ixs = [i for i, x in enumerate(scores) if x == os]
        for i in ixs:
            bert_labels.append(labels[i])
    return bert_labels


def get_metrics(labels_groups, scores_groups):

    map_labels_sum, map_bert_sum = 0, 0
    R_prec_labels_sum, R_prec_bert_sum,  = 0, 0
    recip_rank_labels_sum, recip_rank_bert_sum = 0, 0
    precision_20_labels_sum, precision_20_bert_sum = 0, 0
    ndcg_20_labels_sum, ndcg_20_bert_sum = 0, 0

    for i in zip(labels_groups, scores_groups):

        labels, scores = i[0], i[1]
        bert_labels = get_bert_labels(labels=labels, scores=scores)

        map_labels_sum += get_map(run=labels)
        map_bert_sum += get_map(run=bert_labels)

        R_prec_labels_sum += get_R_prec(run=labels)
        R_prec_bert_sum += get_R_prec(run=bert_labels)

        recip_rank_labels_sum += get_recip_rank(run=labels)
        recip_rank_bert_sum += get_recip_rank(run=bert_labels)

        precision_20_labels_sum += get_precision(run=labels, k=20)
        precision_20_bert_sum += get_precision(run=bert_labels, k=20)

        ndcg_20_labels_sum += get_ndcg(run=labels, k=20)
        ndcg_20_bert_sum += get_ndcg(run=bert_labels, k=20)

    num_queries = len(labels_groups)

    map_labels, map_bert = map_labels_sum / num_queries, map_bert_sum / num_queries
    R_prec_labels, R_prec_bert = R_prec_labels_sum / num_queries, R_prec_bert_sum / num_queries
    recip_rank_labels, recip_rank_bert = recip_rank_labels_sum / num_queries, recip_rank_bert_sum / num_queries
    precision_20_labels, precision_20_bert = precision_20_labels_sum / num_queries, precision_20_bert_sum / num_queries
    ndcg_20_labels, ndcg_20_bert = ndcg_20_labels_sum / num_queries, ndcg_20_bert_sum / num_queries

    string_labels = ['map', 'R_prec', 'recip_rank', 'precision_20', 'ndcg_20']
    label_metrics = [map_labels, R_prec_labels, recip_rank_labels, precision_20_labels, ndcg_20_labels]
    bert_metrics = [map_bert, R_prec_bert, recip_rank_bert, precision_20_bert, ndcg_20_bert]

    return string_labels, label_metrics, bert_metrics


def group_bert_outputs_by_query(label_list, score_list, query_docids_map):

    last_query = 'Not a query'
    labels_groups, scores_groups, queries_groups, doc_ids_groups = [], [], [], []
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

    return labels_groups, scores_groups, queries_groups, doc_ids_groups


def write_trec_run(scores_groups, queries_groups, doc_ids_groups, write_path):

    with open(write_path, "a+") as f:
        for i in zip(scores_groups, queries_groups, doc_ids_groups):

            queries = i[1]
            assert len(set(queries)) == 1, 'Too many queries: {}'.format(queries)

            query = queries[0]
            scores = i[0]
            doc_ids = i[2]

            d = {i[0]: i[1] for i in zip(doc_ids, scores)}
            od = collections.OrderedDict(sorted(d.items(), key=lambda item: item[1], reverse=True))
            rank = 1
            for doc_id in od.keys():
                output_line = " ".join((query, "Q0", str(doc_id), str(rank), str(od[doc_id]), "BERT")) + '\n'
                f.write(output_line)
                rank += 1


if __name__ == '__main__':

    run1 = [1,0,0]
    run2 = [1, 0, 1, 0, 1]
    run3 = [0, 0, 0, 0]
    run4 = [0, 0, 1, 0, 1]

    for r in [run1, run2, run3, run4]:
        print('--------------------------')
        print(r)
        map = get_map(r)
        print('map: {}'.format(map))
        R_prec = get_R_prec(r)
        print('R_prec: {}'.format(R_prec))
        recip_rank = get_recip_rank(r)
        print('recip_rank: {}'.format(recip_rank))
        k = 4
        precision = get_precision(r, k=k)
        print('precision@{}: {}'.format(k, precision))
        ndcg = get_ndcg(r, k=k)
        print('ndcg@{}: {}'.format(k, ndcg))





