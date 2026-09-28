import math
from collections import defaultdict

def query_to_vector(query_tokens: list[str], idf: dict) -> dict[str, float]:
    tf = defaultdict(int)
    for token in query_tokens:
        tf[token] += 1
    
    q_vec = {}
    for token, count in tf.items():
        if token in idf:
            q_vec[token] = count * idf[token]
    return q_vec

def vector_search(query_vector: dict[str, float], inverted_index: dict, idf: dict, doc_lengths: dict, top_k: int = 10):
    if not query_vector:
        return []

    q_norm = math.sqrt(sum(w ** 2 for w in query_vector.values()))
    if q_norm == 0:
        return []

    scores = defaultdict(float)
    for term, q_weight in query_vector.items():
        if term in inverted_index:
            for doc_id, tf in inverted_index[term].items():
                doc_weight = tf * idf[term]
                scores[doc_id] += q_weight * doc_weight

    results = []
    for doc_id, dot_product in scores.items():
        sim = dot_product / (q_norm * doc_lengths[doc_id])
        results.append((doc_id, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def rocchio_update(query_vector: dict[str, float], relevant_doc_ids: list[str], 
                   doc_vectors: dict[str, dict[str, float]], alpha: float = 1.0, beta: float = 0.75):

    new_vector = defaultdict(float)

    for term, weight in query_vector.items():
        new_vector[term] += alpha * weight

    if relevant_doc_ids:
        weight_factor = beta / len(relevant_doc_ids)
        for doc_id in relevant_doc_ids:
            if doc_id in doc_vectors:
                for term, weight in doc_vectors[doc_id].items():
                    new_vector[term] += weight_factor * weight

    return {term: weight for term, weight in new_vector.items() if weight > 0.01}