
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
        r_run = run[:R]
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
    return 0.0


def get_ndcg_cut(run, k=20):
    k_run = run[:k]
    rank = 1
    i_dcg = 0
    dcg = 0
    for r in k_run:
        i_dcg += 1 / np.log2(rank + 1)
        dcg += r / np.log2(rank + 1)

    return dcg / i_dcg


def get_bert_labels(labels, scores):
    bert_labels = []
    ordered_scores = sorted(list(set(scores)), reverse=True)
    for os in ordered_scores:
        ixs = [i for i, x in enumerate(scores) if x == os]
        for i in ixs:
            bert_labels.append(labels[i])
    return bert_labels


def get_stats(labels_groups, scores_groups):

    map_labels_sum = 0
    map_bert_sum = 0

    for i in zip(labels_groups, scores_groups):
        bert_labels = get_bert_labels(labels=i[0], scores=i[1])

        map_labels_sum += get_map(run=i[0])
        map_bert_sum += get_map(run=bert_labels)

    map_labels = map_labels_sum / len(labels_groups)
    map_bert = map_bert_sum / len(labels_groups)

    return map_labels, map_bert


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


# def split_bert_outputs(label_list, score_list, query_docids_map):
#
#     last_query = 'Not a query'
#     doc_counter = 0
#     labels = []
#     scores = []
#     query_counter = 0
#     map_labels_sum = 0
#     map_bert_labels_sum = 0
#
#     for i in zip(label_list, score_list, query_docids_map):
#         query = i[2][0]
#         if (doc_counter > 0) and (last_query != query):
#
#             map_labels, map_bert_labels = get_stats(labels=labels, scores=scores)
#             query_counter += 1
#             map_labels_sum += map_labels
#             map_bert_labels_sum += map_bert_labels
#
#             doc_counter = 0
#             labels = []
#             scores = []
#
#         labels.append(i[0])
#         scores.append(i[1])
#         last_query = query
#         doc_counter += 1
#
#     map_labels, map_bert_labels = get_stats(labels=labels, scores=scores)
#     map_labels_sum += map_labels
#     map_bert_labels_sum += map_bert_labels
#     query_counter += 1
#
#     return (map_labels_sum / query_counter), (map_bert_labels_sum / query_counter)

# scores
# lables
# query
# doc_id


def write_trec_run(score_list, query_docids_map):

    last_query = 'Not a query'
    doc_counter = 0
    scores = []
    doc_ids = []

    for i in zip(score_list, query_docids_map):
        query = i[0][0]
        if (doc_counter > 0) and (last_query != query):

            doc_counter = 0
            scores = []
            doc_ids = []

        scores.append(i[0])
        doc_ids.append(i[1][1])
        last_query = query
        doc_counter += 1

    scores.append(i[0])
    doc_ids.append(i[1][1])





    # d = {i[0]: i[1] for i in zip(doc_ids, scores)}
    # od = collections.OrderedDict(sorted(d.items(), key=lambda item: item[1], reverse=True))
    #
    # run_file
    # rank = 1
    # for doc_id in od.keys():
    #
    #     output_line = " ".join((query, "Q0", str(doc_id), str(rank), str(od[doc_id]), "BERT"))
    #     run_file.write(output_line + "\n")
    #     rank += 1




# possible_write = len(pred_list) // num_rank
# while counter_written < possible_write:
#
#     start_idx = counter_written * num_rank
#     end_idx = counter_written * num_rank + num_rank
#
#     scores = pred_list[start_idx:end_idx]
#     query_docids = query_docids_map[start_idx:end_idx]
#
#     queries, doc_ids = zip(*query_docids)
#     assert len(set(queries)) == 1, "Queries must be all the same. \n queries: {} \n doc_ids: {}".format(queries, doc_ids)
#     assert len(query_docids) == len(scores) == num_rank, 'not correct dimensions'
#     query = queries[0]
#     print(query)
#
#     d = {i[0]:i[1] for i in zip(doc_ids, scores)}
#     od = collections.OrderedDict(sorted(d.items(), key=lambda item: item[1], reverse=True))
#
#     rank = 1
#     for doc_id in od.keys():
#
#         output_line = " ".join((query, "Q0", str(doc_id), str(rank), str(od[doc_id]), "BERT"))
#         run_file.write(output_line + "\n")
#         rank += 1
#
#     counter_written += 1









