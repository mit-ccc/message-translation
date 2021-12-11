#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

#################################
# BERTweet (pre-trained)
#################################

data_folder=blm_alm/raw
for group_name in anti_blm_100k pro_blm_200k 
do
    for vocab_targets in shared_vocab
    do
        # collect embeddings
        python ./code/collect_roberta.py ./models/bertweet/config.json \
            ../data/${data_folder}/${group_name}.txt \
            ../data/${data_folder}/targets/${vocab_targets}.txt \
            ../data/${data_folder}/matrices/matrix_${group_name}_${vocab_targets}.dict.pickle \
            --batch 256 --context 64
    done
done


data_folder=blm_alm/raw
mkdir ../data/${data_folder}/results
for num_layer in 1 2 3 4 5 6 7 8 9 10 11 12
do
    for vocab_targets in shared_vocab
    do
        # Compute cosine distance
        python ./code/dist_cosine.py \
        --num_layers ${num_layer} \
        --target ../data/${data_folder}/targets/${vocab_targets}.txt \
        --input0 ../data/${data_folder}/matrices/matrix_anti_blm_100k_${vocab_targets}.dict.pickle \
        --input1 ../data/${data_folder}/matrices/matrix_pro_blm_200k_${vocab_targets}.dict.pickle \
        --output ../data/${data_folder}/results/${data_name}/${vocab_targets}_last_layer${num_layer}_anti_blm_100k_pro_blm_200k_cos_dist.csv
    done
done









