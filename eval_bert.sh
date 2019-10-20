cd relogic
python -u -m relogic.main \
    --task_name pairwise \
    --mode eval \
    --local_rank $1 \
    --qrels_file_path ../data/zh_en/test \
    --restore_path saves/pair_matching/$2 
