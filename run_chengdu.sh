#!/usr/bin/bash

option=""
dataset=chengdu
max_size=10000
data_name=${max_size}
latlon_config=chengdu.json
location_threshold=0
time_threshold=0
n_bins=62
seed_for_dataset=0
startend=True
training_data_name=bin${n_bins}_seed${seed_for_dataset}_startend${startend}
if [ ${startend} = True ]; then
    option=--startend
fi
python3 make_raw_data.py --original_data_name $dataset --max_size $max_size --seed $seed_for_dataset --save_name $data_name --n_bins $n_bins
python3 data_pre_processing.py --latlon_config  $latlon_config --dataset $dataset --data_name $data_name --location_threshold $location_threshold --time_threshold $time_threshold --save_name $training_data_name --n_bins $n_bins $option --seed $seed_for_dataset
./exp/naive.sh 0 $dataset $data_name $training_data_name > ./exp/log/naive_chengdu.log &
./exp/pre.sh 1 $dataset $data_name $training_data_name > ./exp/log/pre_chengdu.log &
./exp/tree.sh 2 $dataset $data_name $training_data_name > ./exp/log/tree_chengdu.log & 
./exp/pre_tree.sh 3 $dataset $data_name $training_data_name > ./exp/log/pre_tree_chengdu.log &
wait
./exp/pre_tree_single.sh 0 $dataset $data_name $training_data_name > ./exp/log/pre_tree_single_chengdu.log &
./exp/tree_single.sh 1 $dataset $data_name $training_data_name > ./exp/log/tree_single_chengdu.log &
./exp/laplace.sh 3 $dataset $data_name $training_data_name > ./exp/log/laplace_chengdu.log &