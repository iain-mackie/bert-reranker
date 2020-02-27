
import collections
import numpy as np

#TODO - test metrics

def get_map(run):
    correct_docs_sum = sum(run)
    if correct_docs_sum > 0.0:
        precision_sum = 0
        for i, r in enumerate(run):
            if r == 1.0:
                precision_sum += get_precision(run=run, k=(i+1))
        return precision_sum / correct_docs_sum
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


def get_ndcg(run, k=20):
    k_run = run[:k]
    i_dcg = 0
    dcg = 0
    num_rel = sum(run)
    if num_rel > 0:
        for i, r in enumerate(k_run):
            if i == 0:
                if (i+1) <= num_rel:
                    i_dcg += 1
                dcg += r
            else:
                discount = np.log2(i+1)
                if (i+1) <= num_rel:
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

    map_labels_sum, map_bert_sum = 0, 0
    R_prec_labels_sum, R_prec_bert_sum,  = 0, 0
    recip_rank_labels_sum, recip_rank_bert_sum = 0, 0
    precision_20_labels_sum, precision_20_bert_sum = 0, 0
    ndcg_20_labels_sum, ndcg_20_bert_sum = 0, 0
    recall_40_labels_sum, recall_40_bert_sum = 0, 0

    for i in zip(labels_groups, scores_groups, rel_docs_groups):

        labels, scores, R = i[0], i[1], i[2][0]
        bert_labels = get_bert_labels(labels=labels, scores=scores)

        map_labels_sum += get_map(run=labels)
        map_bert_sum += get_map(run=bert_labels)
        print('-------------')
        print(get_map(run=labels))
        print('map: {0:.4f}'.format(get_map(run=labels)))

        R_prec_labels_sum += get_R_prec(run=labels, R=R)
        R_prec_bert_sum += get_R_prec(run=bert_labels, R=R)
        print('R_prec: {0:.4f}'.format(get_R_prec(run=bert_labels, R=R)))

        recip_rank_labels_sum += get_recip_rank(run=labels)
        recip_rank_bert_sum += get_recip_rank(run=bert_labels)
        print('recip_rank: {0:.4f}'.format(get_recip_rank(run=labels)))

        precision_20_labels_sum += get_precision(run=labels, k=20)
        precision_20_bert_sum += get_precision(run=bert_labels, k=20)
        print('precision_20: {0:.4f}'.format(get_precision(run=labels, k=20)))

        recall_40_labels_sum += get_recall(run=labels, k=40, R=R)
        recall_40_bert_sum += get_recall(run=bert_labels, k=40, R=R)
        print('recall_40: {0:.4f}'.format(get_recall(run=labels, k=40, R=R)))

        ndcg_20_labels_sum += get_ndcg(run=labels, k=20)
        ndcg_20_bert_sum += get_ndcg(run=bert_labels, k=20)
        print('ndcg_20: {0:.4f}'.format(get_ndcg(run=labels, k=20)))

    num_queries = len(labels_groups)

    map_labels, map_bert = map_labels_sum / num_queries, map_bert_sum / num_queries
    R_prec_labels, R_prec_bert = R_prec_labels_sum / num_queries, R_prec_bert_sum / num_queries
    recip_rank_labels, recip_rank_bert = recip_rank_labels_sum / num_queries, recip_rank_bert_sum / num_queries
    precision_20_labels, precision_20_bert = precision_20_labels_sum / num_queries, precision_20_bert_sum / num_queries
    ndcg_20_labels, ndcg_20_bert = ndcg_20_labels_sum / num_queries, ndcg_20_bert_sum / num_queries
    recall_40_labels, recall_40_bert = recall_40_labels_sum / num_queries, recall_40_bert_sum / num_queries

    string_labels = ['map', 'R_prec', 'recip_rank', 'precision_20', 'recall_40', 'ndcg_20']
    label_metrics = [map_labels, R_prec_labels, recip_rank_labels, precision_20_labels, recall_40_labels, ndcg_20_labels]
    bert_metrics = [map_bert, R_prec_bert, recip_rank_bert, precision_20_bert, recall_40_bert, ndcg_20_bert]

    return string_labels, label_metrics, bert_metrics


def group_bert_outputs_by_query(label_list, score_list, query_docids_map, query_rel_doc_map=None):

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


if __name__ == '__main__':

    run1 = [1,0,0]
    run2 = [1, 0, 1, 0, 1]
    run3 = [0, 0, 0, 0]
    run4 = [0, 0, 1, 0, 1]
    run5 = [1, 1, 0, 1, 0]


    for r in [run1, run2, run3, run4, run5]:
        print('--------------------------')
        k = 4
        R = 4
        print(r)
        map = get_map(r)
        print('map: {}'.format(map))
        R_prec = get_R_prec(r, R=R)
        print('R_prec: {}'.format(R_prec))
        recip_rank = get_recip_rank(r)
        print('recip_rank: {}'.format(recip_rank))
        precision = get_precision(r, k=k)
        print('precision@{}: {}'.format(k, precision))
        ndcg = get_ndcg(r, k=k)
        print('ndcg@{}: {}'.format(k, ndcg))
        recall = get_recall(r, R=R, k=k)
        print('recall@{}: {}'.format(k, recall))



def code():
    from bert_models import run_metrics
    from bert_utils import build_validation_data_loader
    import torch

    dev_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmarkY1.pt'
    print('loading dev tensor: {}'.format(dev_path))
    validation_tensor = torch.load(dev_path)
    batch_size = 8
    validation_dataloader = build_validation_data_loader(tensor=validation_tensor, batch_size=batch_size)
    run_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmarkY1.run'
    qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmarkY1.qrels'
    run_metrics(validation_dataloader, run_path, qrels_path)
    """
    map                   	all	0.1512
    Rprec                 	all	0.1265
    recip_rank            	all	0.1983
    P_20                  	all	0.0202
    recall_40             	all	0.2521
    ndcg_cut_20           	all	0.1880

    """
