import tensorflow as tf
from include.Model import training, build_HMAN, build_MAN
from include.Test import get_hits
from include.Load import *
import os
import pickle
import argparse

p = argparse.ArgumentParser()
p.add_argument("--lang", help="specify the language pair. (option: zh_en, ja_en, fr_en)")
p.add_argument("--gpu", help="specify the gpu id. (default=0)", default="0")
p.add_argument("--hybrid", help="specify 1=HMAN/0=MAN. (default=1)", default="1")
args = p.parse_args()

LANG = args.lang
GPU = args.gpu
HYBRID = int(args.hybrid)

os.environ["CUDA_VISIBLE_DEVICES"]=GPU
seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

class Config:
    language = LANG # zh_en | ja_en | fr_en
    e1 = 'data/' + language + '/ent_ids_1'
    e2 = 'data/' + language + '/ent_ids_2'
    r1 = 'data/' + language + '/rel_ids_1'
    r2 = 'data/' + language + '/rel_ids_2'
    a1 = 'data/' + language + '/training_attrs_1'
    a2 = 'data/' + language + '/training_attrs_2'
    ill = 'data/' + language + '/ref_ent_ids'
    tr = 'data/' + language + '/train'
    te = 'data/' + language + '/test'
    dev = 'data/' + language + '/dev'
    kg1 = 'data/' + language + '/triples_1'
    kg2 = 'data/' + language + '/triples_2'
    epochs = 50000 if HYBRID else 2000
    se_dim = 200
    ae_dim = 100
    attr_num = 1000
    rel_dim = 100
    rel_num = 1000
    act_func = tf.nn.relu
    gamma = 3.0  # margin based loss
    k = 25  # number of negative samples for each positive one
    ckpt = "ckpt"

if __name__ == '__main__':
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))
    print(e)
    ILL = loadfile(Config.ill, 2)
    illL = len(ILL)
    train = loadfile(Config.tr, 2)
    dev = loadfile(Config.dev, 2)
    np.random.shuffle(train)
    np.random.shuffle(dev)
    train = np.array(train + dev)
    test = loadfile(Config.te, 2)
    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)
    ent2id = get_ent2id([Config.e1, Config.e2]) # attr
    attr = load_attr([Config.a1, Config.a2], e, ent2id, Config.attr_num) # attr
    rel = load_relation(e, KG1+KG2, Config.rel_num)

    if HYBRID:
        print("running HMAN...")
        output_layer, loss = build_HMAN(Config.se_dim, Config.act_func, Config.gamma, Config.k, \
                                        e, train, KG1 + KG2, attr, Config.ae_dim, rel, Config.rel_dim)
    else:
        print("running MAN...")
        output_layer, loss = build_MAN(Config.se_dim, Config.act_func, Config.gamma, Config.k, \
                                        e, train, KG1 + KG2, attr, Config.ae_dim, rel, Config.rel_dim)

    graph_embd, J = training(output_layer, loss, 25, Config.epochs, train, e, Config.k, test)
    get_hits(graph_embd, test)
    with open(Config.ckpt+"/%s_graph_embd.pkl"%Config.language, "wb") as f:
        pickle.dump(graph_embd, f)



