# Hybrid Multi-Aspect Alignment Networks (HMAN)


The code for the research paper [Aligning Cross-lingual Entities with Multi-Aspect Information](https://cs.uwaterloo.ca/~jimmylin/publications/YangHW_etal_EMNLP2019.pdf) @EMNLP 2019. 

*Our code is built on top of [GCN-Align](https://github.com/1049451037/GCN-Align).

## Quick Demo
We stored the embeddings for demo in the directory `demo_embd`, and we can evaluate `ZH-EN` as follows:
```
python weighted_concat.py -d demo_embd/pairwise_dump.json -g demo_embd/zh_en_graph_embd.pkl -i data/zh_en/test
```


## Run Graph-based Embeddings (HMAN/MAN)

#### HMAN
```
bash graph.sh 0 1
```
#### MAN
```
bash graph.sh 0 0
```

## Run PairwiseBERT
#### Training
```
bash train_bert.sh 0 zh_en
```
(Note that you need to stop the training manually.)

#### Evaluation
```
bash eval_bert.sh 0 zh_en
```

## Integration
```
python weighted_concat.py --desc relogic/saves/pair_matching/zh_en/pairwise_dump.json --graph graph_ckpt/zh_en_graph_embd.pkl --ill data/zh_en/test
```


## Citation
```
@article{yang2019align,
  title={Aligning Cross-Lingual Entities with Multi-Aspect Information},
  author={Yang, Hsiu-Wei and Zou, Yanyan and Shi, Peng and Lu, Wei and Lin, Jimmy and Sun, Xu},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year={2019}
}
```

