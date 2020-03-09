
import collections
import random


def get_query_docids_map(run_path):

    query_docids_map = []
    with open(run_path) as ref_file:

        for line in ref_file:
            query, _, doc_id, _, _, _ = line.strip().split(" ")

            query_docids_map.append((query, doc_id))

    return query_docids_map


def get_query_rel_doc_map(qrels_path):

    query_rel_doc_map = {}
    with open(qrels_path) as qrels_file:

        for line in qrels_file:
            query, _, doc_id, _ = line.strip().split(" ")

            if query in query_rel_doc_map:
                query_rel_doc_map[query].append(doc_id)
            else:
                query_rel_doc_map[query] = [doc_id]

    return query_rel_doc_map


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


def write_trec_eval(write_path, label_string, oracle_string, bert_string):

    eval_path = write_path + '.eval'

    with open(eval_path, "a+") as f:
        for s in [label_string, oracle_string, bert_string]:
            f.write(s + '\n')


def get_re_ranking_dict(re_ranking_path):

    re_ranking_dict = collections.OrderedDict()
    with open(re_ranking_path) as re_ranking_file:
        for line in re_ranking_file:
            q, _, doc_id, _, _, _ = line.strip().split(" ")
            if q not in re_ranking_dict:
                re_ranking_dict[q] = []
            re_ranking_dict[q].append(doc_id)
    return re_ranking_dict


def get_reduce_run_dict(run_path, reduce_docs):

    reduce_run_dict = collections.OrderedDict()
    with open(run_path) as run_file:
        for line in run_file:
            q, _, doc_id, r, _, _ = line.strip().split(" ")

            if q not in reduce_run_dict:
                reduce_run_dict[q] = []
            if int(r) <= reduce_docs:
                reduce_run_dict[q].append(doc_id)

    return reduce_run_dict


def reduce_re_ranking_by_original_run(run_path, re_ranking_path, write_path, reduce_docs=10):

    re_ranking_dict = get_re_ranking_dict(re_ranking_path=re_ranking_path)
    reduce_run_dict = get_reduce_run_dict(run_path=run_path, reduce_docs=reduce_docs)

    with open(write_path, "a+") as f:
        for query, docs_ids in re_ranking_dict.items():
            rank = 1
            score = 0.0
            for doc_id in docs_ids:
                if doc_id in reduce_run_dict[query]:
                    output_line = " ".join((query, "Q0", str(doc_id), str(rank), str(score), "BERT-REDUCED")) + '\n'
                    f.write(output_line)
                    rank += 1
                    score -= 0.1


def write_qrels(random_query_od, data_dir, set_name, num_queries):
    write_path = data_dir + set_name + '_' + str(num_queries) + '_random_queries.qrels'
    print('writing qrels to: {}'.format(write_path))
    with open(write_path, "a+") as f:
        for _, lines in random_query_od.items():
            for line in lines:
                f.write(line)


def write_topics(random_query_od, data_dir, set_name, num_queries):
    write_path = data_dir + set_name + '.topics'
    print('writing topics to: {}'.format(write_path))
    with open(write_path, "a+") as f:
        for query, _ in random_query_od.items():
            f.write(query + '\n')


def get_random_queries(queries, num_queries):
    random.shuffle(queries)
    random_queries = []
    for q in queries:
        random_queries.append(q)
        if len(random_queries) == num_queries:
            return sorted(random_queries)


def random_sample_qrels(data_dir, set_name, num_queries):
    qrels_path = data_dir + set_name + '.qrels'
    query_od = collections.OrderedDict()
    with open(qrels_path) as qrels_file:
        for line in qrels_file:
            query, i, doc_id, rank = line.strip().split(" ")

            if query not in query_od:
                query_od[query] = []

            query_od[query].append(line)

    queries = list(query_od.keys())
    max_num_queries = len(queries)
    print(queries)
    if max_num_queries <= num_queries:
        print('Not enough queries to meet request')
        return query_od

    random_queries = get_random_queries(queries=queries, num_queries=num_queries)
    print(random_queries)

    random_query_od = collections.OrderedDict()
    for q in random_queries:
        random_query_od[q] = query_od[q]

    write_topics(random_query_od=random_query_od, data_dir=data_dir, set_name=set_name, num_queries=num_queries)

    write_qrels(random_query_od=random_query_od, data_dir=data_dir, set_name=set_name, num_queries=num_queries)





if __name__ == '__main__':
    # reduce_docs = 100
    # run_path = '/Users/iain/LocalStorage/coding/github/bert-reranker/test_data/test.run'
    # re_ranking_path = '/Users/iain/LocalStorage/coding/github/bert-reranker/test_data/bert_predictions_test.run'
    # write_path = '/Users/iain/LocalStorage/coding/github/bert-reranker/test_data/test_reduced_{}.run'.format(reduce_docs)
    # reduce_re_ranking_by_original_run(run_path=run_path, re_ranking_path=re_ranking_path, write_path=write_path,
    #                                   reduce_docs=reduce_docs)


    num_queries = 500
    data_dir = '/nfs/trec_car/data/bert_reranker_datasets/training_data_sample_queries/'
    set_name = 'train_fold_0_train_hierarchical'
    random_sample_qrels(data_dir, set_name, num_queries)
