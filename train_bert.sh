# Preprocessing
python gen_candidate.py --ckpt graph_ckpt
mkdir -p relogic/data/raw_data/entity_alignment/
cp candidate/zh_en/dev.json relogic/data/raw_data/entity_alignment/dev.json
cp candidate/zh_en/test.json relogic/data/raw_data/entity_alignment/test.json
shuf candidate/zh_en/train.json | head -n 300000 - > relogic/data/raw_data/entity_alignment/train.json

# Training
cd relogic
python -u -m relogic.main \
    --task_name pairwise \
    --mode train \
    --output_dir saves/pair_matching/$2 \ # project name
    --bert_model bert-base-multilingual-cased \
    --raw_data_path data/raw_data/entity_alignment \
    --label_mapping_path none \
    --model_name default \
    --local_rank $1 \ # GPU id
    --train_batch_size 3 \
    --test_batch_size 3 \
    --learning_rate 1e-6 \
    --epoch_number 2 \
    --lang zh \
    --eval_dev_every 3000 \
    --max_seq_length 250 \
    --qrels_file_path ../data/zh_en/dev \


