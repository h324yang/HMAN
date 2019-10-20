mkdir graph_ckpt
python run.py --lang zh_en --gpu $1 --hybrid $2
python run.py --lang ja_en --gpu $1 --hybrid $2
python run.py --lang fr_en --gpu $1 --hybrid $2

