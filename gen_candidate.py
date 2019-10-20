import pickle
from sklearn.metrics import pairwise_distances
from include.Load import loadfile
import numpy as np
import argparse
import os
import json



def load_comment(lang):
    res = {}
    for idx in [1, 2]:
        input_path = "data/" + lang + "/comment_%d"%idx
        with open(input_path) as f:
            for line in f:
                tup = line.strip().split("\t")
                try:
                    res[int(tup[0])] = tup[1]
                except:
                    pass
    return res


class Mode:
    TRAIN = "TRAIN" # must include positive pairs for training
    EVAL = "EVAL"


def write_gcn_dataset(lang, vec, train, dev, test, sent_dict, out_dir, topk=200):
    ref = np.array(loadfile("data/"+lang+"/ref_ent_ids", 2))
    Ls = ref[:,0]
    Rs = ref[:,1]
    Lvec = vec[Ls]
    Rvec = vec[Rs]
    sim = pairwise_distances(Lvec, Rvec, metric='cityblock', n_jobs=10)
    L2row = {left:row for row, left in enumerate(Ls)}
    R2row = {right:row for row, right in enumerate(Rs)}

    def write_train_cand(path, ill):
        with open(path, "w") as f:
            iLs = ill[:,0]
            iRs = ill[:,1]
            iLrow = np.array([L2row[i] for i in iLs])
            iRrow = np.array([R2row[i] for i in iRs])
            internal = sim[iLrow]
            internal = internal.transpose()[iRrow].transpose() # rerank among given ILLs
            cand_for_L = iRs[np.argsort(internal)[:,:topk]]
            cand_for_R = iLs[np.argsort(internal.transpose())[:,:topk]]
            iL2row = {left:row for row, left in enumerate(iLs)}
            iR2row = {right:row for row, right in enumerate(iRs)}

            for tup in ill:
                Lb = cand_for_L[iL2row[tup[0]]]
                Rb = cand_for_R[iR2row[tup[1]]]
                Llabel = Lb == tup[1]
                Rlabel = Rb == tup[0]

                if not tup[1] in Lb:
                    Lb = np.concatenate([Lb[:-1], [np.array(tup[1])]])
                    Llabel = np.concatenate([Llabel[:-1], [np.array(True)]])
                if not tup[0] in Rb:
                    Rb = np.concatenate([Rb[:-1], [np.array(tup[0])]])
                    Rlabel = np.concatenate([Rlabel[:-1], [np.array(True)]])

                for r, label in zip(Lb, Llabel):
                    if not label:
                        example = {"guid":str(tup[0]), "text":sent_dict[tup[0]],
                                   "p_guid":str(tup[1]), "text_p":sent_dict[tup[1]],
                                   "n_guid":str(r), "text_n":sent_dict[r]}
                        f.write(json.dumps(example)+"\n")

                for l, label in zip(Rb, Rlabel):
                    if not label:
                        example = {"guid":str(tup[1]), "text":sent_dict[tup[1]],
                                   "p_guid":str(tup[0]), "text_p":sent_dict[tup[0]],
                                   "n_guid":str(l), "text_n":sent_dict[l]}
                        f.write(json.dumps(example)+"\n")

    def write_test_sent(path, ill):
        with open(path, "w") as f:
            for tup in ill:
                example = {"guid":str(tup[0]), "text":sent_dict[tup[0]]}
                f.write(json.dumps(example)+"\n")
                example = {"guid":str(tup[1]), "text":sent_dict[tup[1]]}
                f.write(json.dumps(example)+"\n")

    if not os.path.isdir(out_dir+"/"+lang):
        os.makedirs(out_dir+"/"+lang)
    write_train_cand(out_dir+"/"+lang+"/train.json", train)
    write_test_sent(out_dir+"/"+lang+"/dev.json", dev)
    write_test_sent(out_dir+"/"+lang+"/test.json", test)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--lang") # {zh_en, ja_en, fr_en}
    args = p.parse_args()

    comm_dict = load_comment(args.lang)
    vec = pickle.load(open(args.ckpt+"/"+args.lang+"_graph_embd.pkl", "rb"))
    train = np.array(loadfile("data/"+args.lang+"/train", 2))
    dev = np.array(loadfile("data/"+args.lang+"/dev", 2))
    test = np.array(loadfile("data/"+args.lang+"/ref_ent_ids", 2)) # to get all desc embeddings
    write_gcn_dataset(args.lang, vec, train, dev, test, comm_dict, "candidate", topk=200)


