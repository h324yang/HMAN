import random
import argparse
import numpy as np
import scipy.spatial
from include.Test import get_hits
from include.Load import *
import pickle


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--desc", help="Paht to PairwiseBERT embedding file. (JSON)")
    p.add_argument("-g", "--graph", help="Paht to graph embedding. (PICKLE)")
    p.add_argument("-i", "--ill", help="Paht to ILLs, i.e., ground truth. (TXT)")
    args = p.parse_args()
    return args

def main():
    args = parse()
    ill = loadfile(args.ill, 2)
    bert_dict = load_json_embd(args.desc)
    with open(args.graph, "rb") as f:
        graph_embd = pickle.load(f)
    e_num, _ = graph_embd.shape
    bert_embd = np.array([bert_dict[i] if i in bert_dict else np.zeros_like(bert_dict[0]) for i in range(e_num)])
    embd = np.concatenate([0.8*graph_embd, 0.2*bert_embd], axis=1)
    get_hits(embd, ill)


if __name__ == "__main__":
    main()
