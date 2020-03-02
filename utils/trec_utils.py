
import collections

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