import numpy as np
import scipy


def get_hits(vecs, test_pairs, top_k=(1, 10, 50)):
    head_vecs = np.array([vecs[e1] for e1, e2 in test_pairs])
    tail_vecs = np.array([vecs[e2] for e1, e2 in test_pairs])
    sim = scipy.spatial.distance.cdist(head_vecs, tail_vecs, metric='cityblock')
    hit_k_linked = [0] * len(top_k)
    hit_k_nil = [0] * len(top_k)
    for i, pair in enumerate(test_pairs):
        ranked_tail_ids = [test_pairs[idx][1] for idx in sim[i, :].argsort()]
        target_rank = ranked_tail_ids.index(pair[1])
        for j in range(len(top_k)):
            if target_rank < top_k[j]:
                hit_k = hit_k_nil if pair[1] == "-1" else hit_k_linked
                hit_k[j] += 1
                
    hit_k_all = [hitl + hitn for hitl, hitn in zip(hit_k_linked, hit_k_nil)]
    
    results = []
    total_all = len(test_pairs)
    total_linked = total_all - ranked_tail_ids.count("-1")
    total_nil = total_all - total_linked
    for hit_k, total, category in [
        (hit_k_all, total_all, "All"), 
        (hit_k_linked, total_linked, "Linked"), 
        (hit_k_nil, total_nil, "NIL"), 
    ]:
        if not total:
            print(f"No result for {category}...")
            continue
            
        for i in range(len(hit_k)):
            res = hit_k[i]/total*100
            print(f"Result (#{category}: {total}) | Hits@{top_k[i]}: {res:.2f}%")
            results.append((category, total, top_k[i], res))
        
    return results
