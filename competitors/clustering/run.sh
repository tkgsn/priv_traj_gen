#!/bin/bash

data_size=$MAX_SIZE
dataset=$DATASET
n_bins=$N_BINS
epsilon=$EPSILON
k=$K
seed=$SEED
# data_size=0
# dataset=geolife_mm
# n_bins=30


training_data_name=200_30_bin${n_bins}_seed0

# training_data_dir=/data/${dataset}/${data_size}/${training_data_name}
# save_name=generalization

# ks=(5 10 15 20 25 30 35 40 45 50 100 150 200 250 300)
# ks=(10)
# epsilons=(0.1 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5)
# ks=(100)
# iterate over k and epsilon
# for k in ${ks[@]}
# do
python3 run.py --dataset $dataset --data_name $data_size --training_data_name $training_data_name --k $k --epsilon $epsilon --seed $SEED
# done