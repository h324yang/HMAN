import numpy as np
from collections import Counter

# load a file and return a list of tuple containing $num integers in each line
def loadfile(fn, num=1):
        print('loading a file...' + fn)
        ret = []
        with open(fn, encoding='utf-8') as f:
                for line in f:
                        th = line[:-1].split('\t')
                        x = []
                        for i in range(num):
                                x.append(int(th[i]))
                        ret.append(tuple(x))
        return ret


def get_ent2id(fns):
        ent2id = {}
        for fn in fns:
                with open(fn, 'r', encoding='utf-8') as f:
                        for line in f:
                                th = line[:-1].split('\t')
                                ent2id[th[1]] = int(th[0])
        return ent2id


# The most frequent attributes are selected to save space
def load_attr(fns, e, ent2id, topA=1000):
        cnt = {}
        for fn in fns:
                with open(fn, 'r', encoding='utf-8') as f:
                        for line in f:
                                th = line[:-1].split('\t')
                                if th[0] not in ent2id:
                                        continue
                                for i in range(1, len(th)):
                                        if th[i] not in cnt:
                                                cnt[th[i]] = 1
                                        else:
                                                cnt[th[i]] += 1
        fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
        attr2id = {}
        for i in range(topA):
                attr2id[fre[i][0]] = i
        attr = np.zeros((e, topA), dtype=np.float32)
        for fn in fns:
                with open(fn, 'r', encoding='utf-8') as f:
                        for line in f:
                                th = line[:-1].split('\t')
                                if th[0] in ent2id:
                                        for i in range(1, len(th)):
                                                if th[i] in attr2id:
                                                        attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
        return attr



def load_relation(e, KG, topR=1000):
        rel_mat = np.zeros((e, topR), dtype=np.float32)
        rels = np.array(KG)[:,1]
        top_rels = Counter(rels).most_common(topR)
        rel_index_dict = {r:i for i,(r,cnt) in enumerate(top_rels)}
        for tri in KG:
                h = tri[0]
                r = tri[1]
                o = tri[2]
                if r in rel_index_dict:
                        rel_mat[h][rel_index_dict[r]] += 1.
                        rel_mat[o][rel_index_dict[r]] += 1.
        return np.array(rel_mat)


