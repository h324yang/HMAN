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
tf.compat.v1.set_random_seed(seed)

class Config:
    language = LANG # zh_en | ja_en | fr_en
    wikidata = True
    data_dir = '../massive-align/WikidataLow/data/'
    e1 = data_dir + language + '/ent_ids_1'
    e2 = data_dir + language + '/ent_ids_2'
    r1 = data_dir + language + '/rel_ids_1'
    r2 = data_dir + language + '/rel_ids_2'
    a1 = data_dir + language + '/training_attrs_1'
    a2 = data_dir + language + '/training_attrs_2'
    ill = data_dir + language + '/ref_ent_ids'
    tr = data_dir + language + '/train'
    te = data_dir + language + '/test' # False
    dev = False # data_dir + language + '/dev' 
    kg1 = data_dir + language + '/triples_1'
    kg2 = data_dir + language + '/triples_2'
    epochs = 50000 if HYBRID else 2000
    se_dim = 100
    ae_dim = 50
    rel_dim = 50
    attr_num = 1000
    rel_num = 100
    act_func = tf.nn.relu
    gamma = 3.0  # margin based loss
    k = 25  # number of negative samples for each positive one
    ckpt = "../scratch/graph_ckpt"
    global_nil = True
    enable_nil = True

if __name__ == '__main__':
    e1 = set(loadfile(Config.e1, 1))
    e2 = set(loadfile(Config.e2, 1))
    num_ents = len(e1 | e2)
    e = num_ents
    
    if Config.enable_nil:
        # reindex nil nodes
        postporc = lambda tup: tuple([num_ents if i == -1 else i for i in tup])
        e += 1
    else:
        # remove nil nodes
        postporc = lambda tup: None if -1 in tup else tup

    ILL = loadfile(Config.ill, 2, postporc)
    train = loadfile(Config.tr, 2, postporc)
    np.random.shuffle(train)
    
    if Config.dev:
        dev = loadfile(Config.dev, 2, postporc)
        np.random.shuffle(dev)
        train = np.array(train + dev)
    else:
        train = np.array(train)
    
    if Config.te:
        # -1, the original id of nil, is used while test
        # so if nil is enabled, do nothing to nil entities 
        te_postproc = None if Config.enable_nil else postporc
        test = loadfile(Config.te, 2, te_postproc)
    else: 
        test = False
        
    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)
    ent2id = get_ent2id([Config.e1, Config.e2]) # attr
    attr = load_attr([Config.a1, Config.a2], e, ent2id, Config.attr_num) # attr
    
    if Config.enable_nil and Config.global_nil:
        # nil connects to all entities and has all attributes
        KG2 += [(e-1, 0, int(ent[0])) for ent in e2]
        attr[-1] = np.ones(attr.shape[1])
    
    rel = load_relation(e, KG1+KG2, Config.rel_num)

    print(f"num of total refs: {len(ILL)}")

    if HYBRID:
        print("running HMAN...")
        output_layer, loss = build_HMAN(
            Config.se_dim, Config.act_func, Config.gamma, Config.k,
            e, train, KG1 + KG2, attr, Config.ae_dim, rel, Config.rel_dim,
            Config.wikidata # one-way loss if it's wikidata
        )
    else:
        print("running MAN...")
        output_layer, loss = build_MAN(
            Config.se_dim, Config.act_func, Config.gamma, Config.k,
            e, train, KG1 + KG2, attr, Config.ae_dim, rel, Config.rel_dim,
            Config.wikidata # one-way loss if it's wikidata
        )
    
    # dict output is used if it's wikidata, else 2d-array
    graph_embd, J = training(
        output_layer, loss, 1.0, Config.epochs, train, e, Config.k, test, Config.wikidata
    )
    
    with open(Config.ckpt+"/%s_graph_embd.pkl"%Config.language, "wb") as f:
        pickle.dump(graph_embd, f)



