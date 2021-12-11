#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

###########
### RAW ###
###########

python ./src/stats/counts.py ../data/blm_alm/raw/ ../data/blm_alm/raw/counts.json

python ./src/stats/topics.py ../data/blm_alm/raw/master.model ../data/blm_alm/raw/clusters.json

sh ./src/modeling/multitrain.sh ../data/blm_alm/raw/ ../data/blm_alm/raw/master.model

mkdir ../data/blm_alm/raw/models/
sh ./src/modeling/train.sh ../data/blm_alm/raw/ ../data/blm_alm/raw/models/

for k in 0 500 1000 3000 5000 -1
do
    python3 ./src/alignment/align.py ../data/blm_alm/raw/models/ ../data/blm_alm/raw/aligners/cca/${k} ../data/blm_alm/raw/counts.json cca $k
    python3 ./src/alignment/align.py ../data/blm_alm/raw/models/ ../data/blm_alm/raw/aligners/svd/${k}  ../data/blm_alm/raw/counts.json svd $k
    python3 ./src/alignment/align.py ../data/blm_alm/raw/models/ ../data/blm_alm/raw/aligners/lstsq/${k}  ../data/blm_alm/raw/counts.json lstsq $k
done
