
import collections
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
    i_dcg = 0
    dcg = 0
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


def get_metrics(labels_groups, scores_groups, rel_docs_groups):
    #TODO - refactor
    map_original_sum, map_bert_sum, map_oracle_sum = 0, 0, 0
    R_prec_original_sum, R_prec_bert_sum, R_prec_oracle_sum = 0, 0, 0
    recip_rank_original_sum, recip_rank_bert_sum, recip_rank_oracle_sum = 0, 0, 0
    precision_20_original_sum, precision_20_bert_sum, precision_20_oracle_sum = 0, 0, 0
    ndcg_20_original_sum, ndcg_20_bert_sum, ndcg_20_oracle_sum = 0, 0, 0
    recall_40_original_sum, recall_40_bert_sum, recall_40_oracle_sum = 0, 0, 0

    for i in zip(labels_groups, scores_groups, rel_docs_groups):

        original_rank, scores, R = i[0], i[1], i[2][0]
        bert_rank = get_bert_labels(labels=original_rank, scores=scores)
        oracle_rank = sorted(original_rank, reverse=True)

        map_original_sum += get_map(run=original_rank, R=R)
        map_bert_sum += get_map(run=bert_rank, R=R)
        map_oracle_sum += get_map(run=oracle_rank, R=R)

        R_prec_original_sum += get_R_prec(run=original_rank, R=R)
        R_prec_bert_sum += get_R_prec(run=bert_rank, R=R)
        R_prec_oracle_sum += get_R_prec(run=oracle_rank, R=R)

        recip_rank_original_sum += get_recip_rank(run=original_rank)
        recip_rank_bert_sum += get_recip_rank(run=bert_rank)
        recip_rank_oracle_sum += get_recip_rank(run=oracle_rank)

        precision_20_original_sum += get_precision(run=original_rank, k=20)
        precision_20_bert_sum += get_precision(run=bert_rank, k=20)
        precision_20_oracle_sum += get_precision(run=oracle_rank, k=20)

        recall_40_original_sum += get_recall(run=original_rank, k=40, R=R)
        recall_40_bert_sum += get_recall(run=bert_rank, k=40, R=R)
        recall_40_oracle_sum += get_recall(run=oracle_rank, k=40, R=R)

        ndcg_20_original_sum += get_ndcg(run=original_rank, R=R, k=20)
        ndcg_20_bert_sum += get_ndcg(run=bert_rank, R=R, k=20)
        ndcg_20_oracle_sum += get_ndcg(run=oracle_rank, R=R, k=20)

    num_queries = len(labels_groups)

    map_original, map_bert, map_oracle = map_original_sum / num_queries, map_bert_sum / num_queries, map_oracle_sum / num_queries
    R_prec_original, R_prec_bert, R_prec_oracle = R_prec_original_sum / num_queries, R_prec_bert_sum / num_queries, R_prec_oracle_sum / num_queries
    recip_rank_original, recip_rank_bert, recip_rank_oracle = recip_rank_original_sum / num_queries, recip_rank_bert_sum / num_queries, recip_rank_oracle_sum / num_queries
    precision_20_original, precision_20_bert, precision_20_oracle  = precision_20_original_sum / num_queries, precision_20_bert_sum / num_queries, precision_20_oracle_sum / num_queries
    ndcg_20_original, ndcg_20_bert, ndcg_20_oracle = ndcg_20_original_sum / num_queries, ndcg_20_bert_sum / num_queries, ndcg_20_oracle_sum / num_queries
    recall_40_original, recall_40_bert, recall_40_oracle= recall_40_original_sum / num_queries, recall_40_bert_sum / num_queries, recall_40_oracle_sum / num_queries

    mertric_strings = ['map', 'R_prec', 'recip_rank', 'precision_20', 'recall_40', 'ndcg_20']
    label_metrics = [map_original, R_prec_original, recip_rank_original, precision_20_original, recall_40_original, ndcg_20_original]
    bert_metrics = [map_bert, R_prec_bert, recip_rank_bert, precision_20_bert, recall_40_bert, ndcg_20_bert]
    oracle_metrics = [map_bert, R_prec_bert, recip_rank_bert, precision_20_bert, recall_40_bert, ndcg_20_bert]

    return mertric_strings, label_metrics, bert_metrics, oracle_metrics


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


def get_metrics_string(string_labels, metrics, name='BERT'):
    s = '  Average {}:  '.format(name)
    for i in zip(string_labels, metrics):
        s += i[0] + ': {0:.4f}, '.format(i[1])
    return s

def get_results_string(labels, scores):
    s = '  '
    for i in zip(labels, scores):
        s += '(truth: {:.3f}, pred: {:.3f}), '.format(i[0], i[1])
    return s

if __name__ == '__main__':
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
