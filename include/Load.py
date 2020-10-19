import numpy as np
from collections import Counter
import json


# load a file and return a list of tuple containing $num integers in each line
def loadfile(fn, num=1, postproc=None):
        print('loading a file...' + fn)
        ret = []
        with open(fn, encoding='utf-8') as f:
                for line in f:
                        th = line.strip().split('\t')
                        x = []
                        for i in range(num):
                                elem = int(th[i])
                                x.append(elem)
                        ret.append(tuple(x))
        
        if postproc:
            ret = [postproc(tup) for tup in ret if postproc(tup)]
        
        return ret


def get_ent2id(fns, prefix=False):
        ent2id = {}
        for i, fn in enumerate(fns):
                with open(fn, 'r', encoding='utf-8') as f:
                        for line in f:
                                th = line.strip().split('\t')
                                if prefix:
                                    ent2id[f"{i}_{th[1]}"] = int(th[0])
                                else:
                                    ent2id[th[1]] = int(th[0])
        return ent2id


# The most frequent attributes are selected to save space
def load_attr(fns, e, ent2id, topA=1000, prefix=False):
        cnt = {}
        for j, fn in enumerate(fns):
                with open(fn, 'r', encoding='utf-8') as f:
                        for line in f:
                                th = line.strip().split('\t')
                                ent = f"{j}_{th[0]}" if prefix else th[0]
                                if ent not in ent2id:
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
        for j, fn in enumerate(fns):
                with open(fn, 'r', encoding='utf-8') as f:
                        for line in f:
                                th = line.strip().split('\t')
                                ent = f"{j}_{th[0]}" if prefix else th[0]
                                if ent in ent2id:
                                        for i in range(1, len(th)):
                                                if th[i] in attr2id:
                                                        attr[ent2id[ent]][attr2id[th[i]]] = 1.0
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


def load_json_embd(path):
        embd_dict = {}
        with open(path) as f:
                for line in f:
                        example = json.loads(line.strip())
                        vec = np.array([float(e) for e in example['feature'].split()])
                        embd_dict[int(example['guid'])] = vec
        return embd_dict


