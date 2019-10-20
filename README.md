# Hybrid Multi-Aspect Alignment Networks (HMAN)


The code for the research paper [Aligning Cross-lingual Entities with Multi-Aspect Information](https://cs.uwaterloo.ca/~jimmylin/publications/YangHW_etal_EMNLP2019.pdf) @EMNLP 2019. 

*Our code is built on top of [GCN-Align](https://github.com/1049451037/GCN-Align).

## Run Graph-based Embeddings (HMAN/MAN)

#### HMAN
```
python run.py --lang zh_en --gpu 0 --hybrid 1
```
#### MAN
```
python run.py --lang zh_en --gpu 0 --hybrid 0
```

## Run PairwiseBERT
#### Training
```
bash train_bert.sh 0 1
```
Note that you need to stop the training manually.

#### Evaluation
```
bash eval_bert.sh 0 1
```

## Integration
```
python weighted_concat.py --desc relogic/saves/pair_matching/1/pairwise_dump.json \ 
                          --graph graph_ckpt/zh_en_graph_embd.pkl \
                          --ill data/zh_en/test
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

