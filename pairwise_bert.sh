# python gen_candidate.py --ckpt graph_ckpt
# mkdir -p relogic/data/raw_data/entity_alignment/
# cp candidate/zh_en/train.json relogic/data/raw_data/entity_alignment/dev.json
# cp candidate/zh_en/train.json relogic/data/raw_data/entity_alignment/test.json
# shuf candidate/zh_en/train.json > relogic/data/raw_data/entity_alignment/train.json
cd relogic
sh examples/scripts/pair_matching/pair.sh 0
