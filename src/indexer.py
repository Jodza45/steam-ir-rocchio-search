import math
from collections import defaultdict

def build_index(documents: dict[str, list[str]]):
    num_docs = len(documents)
    inverted_index = defaultdict(dict)
    doc_tf = defaultdict(lambda: defaultdict(int))

    for doc_id, tokens in documents.items():
        for token in tokens:
            doc_tf[doc_id][token] += 1
        for token, count in doc_tf[doc_id].items():
            inverted_index[token][doc_id] = count

    idf = {}
    for term, postings in inverted_index.items():
        df = len(postings)
        idf[term] = math.log(num_docs / df)

    doc_vectors = defaultdict(dict)
    doc_lengths = {}

    for doc_id, terms in doc_tf.items():
        sum_sq = 0.0
        for term, tf in terms.items():
            weight = tf * idf[term]
            doc_vectors[doc_id][term] = weight
            sum_sq += weight ** 2
        doc_lengths[doc_id] = math.sqrt(sum_sq)

    return dict(inverted_index), idf, dict(doc_vectors), doc_lengths