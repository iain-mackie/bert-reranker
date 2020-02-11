
from trec_car.read_data import iter_paragraphs, ParaText, ParaLink #iter_pages, iter_annotations,
import os
import collections
import time

#TODO - make datasets (paragraph + sentence)

paragraphs_path = os.path.join(os.getcwd(), 'dedup.articles-paragraphs.cbor')

test_qrels_path = os.path.join(os.getcwd(), 'test.qrels')
test_run_path = os.path.join(os.getcwd(), 'test.run')


def load_qrels(path=test_qrels_path):
    """Loads qrels into a dict of key: topic, value: list of relevant doc ids."""
    qrels = collections.defaultdict(set)
    with open(path) as f:
        for i, line in enumerate(f):
            topic, _, doc_title, relevance = line.rstrip().split(' ')
            if int(relevance) >= 1:
                qrels[topic].add(doc_title)
            if i % 1000000 == 0:
                print('Loading qrels {}'.format(i))
    return qrels


def load_run(path):
    """Loads run into a dict of key: topic, value: list of candidate doc ids."""

    # We want to preserve the order of runs so we can pair the run file with the
    # TFRecord file.
    run = collections.OrderedDict()
    with open(path) as f:
        for i, line in enumerate(f):
            topic, _, doc_title, rank, _, _ = line.split(' ')
            if topic not in run:
                run[topic] = []
            run[topic].append((doc_title, int(rank)))
            if i % 1000000 == 0:
                print('Loading run {}'.format(i))
    # Sort candidate docs by rank.
    sorted_run = collections.OrderedDict()
    for topic, doc_titles_ranks in run.items():
        sorted(doc_titles_ranks, key=lambda x: x[1])
        doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
        sorted_run[topic] = doc_titles

    return sorted_run


def load_corpus(path=paragraphs_path):
    """Loads TREC-CAR's paraghaphs into a dict of key: title, value: paragraph."""
    corpus = {}
    start_time = time.time()
    APPROX_TOTAL_PARAGRAPHS = 30000000
    with open(path, 'rb') as f:
        for i, p in enumerate(iter_paragraphs(f)):
            para_txt = [elem.text if isinstance(elem, ParaText)
                        else elem.anchor_text
                        for elem in p.bodies]

            corpus[p.para_id] = ' '.join(para_txt)
            if i % 10000 == 0:
                print('Loading paragraph {} of {}'.format(i, APPROX_TOTAL_PARAGRAPHS))
                time_passed = time.time() - start_time
                hours_remaining = (APPROX_TOTAL_PARAGRAPHS - i) * time_passed / (max(1.0, i) * 3600)
                print('Estimated hours remaining to load corpus: {}'.format(hours_remaining))

    return corpus

def merge(qrels, run):
    """Merge qrels and runs into a single dict of key: topic, value: tuple(relevant_doc_ids, candidate_doc_ids)"""
    data = collections.OrderedDict()
    for topic, candidate_doc_ids in run.items():
        data[topic] = (qrels[topic], candidate_doc_ids)
    return data


if __name__ == "__main__":
    test = load_qrels()
    print(test)
    print('hi')

