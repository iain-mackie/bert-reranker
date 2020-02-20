
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


def get_bert_labels(labels, scores):

    bert_labels = []
    ordered_scores = sorted(list(set(scores)), reverse=True)
    for os in ordered_scores:
        ixs = [i for i, x in enumerate(scores) if x == os]
        for i in ixs:
            bert_labels.append(labels[i])
    return bert_labels



def get_stats(labels, scores):
    print('get stats')
    print('labels', labels)
    print('scores', scores)

    bert_labels = get_bert_labels(labels=labels, scores=scores)
    print('bert labels', bert_labels)

    map_labels = get_map(run=labels)
    map_bert_labels = get_map(run=bert_labels)

    return map_labels, map_bert_labels

