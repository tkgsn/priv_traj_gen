#!/usr/bin/bash

option=""
dataset=rotation
max_size=10000
data_name=$max_size
latlon_config=test.json
location_threshold=0
time_threshold=0
n_bins=30
seed_for_dataset=0
training_data_name=bin${n_bins}_seed${seed_for_dataset}
python3 make_raw_data.py --original_data_name $dataset --max_size $max_size --seed $seed_for_dataset --save_name $data_name --n_bins $n_bins
python3 data_pre_processing.py --latlon_config  $latlon_config --dataset $dataset --data_name $data_name --location_threshold $location_threshold --time_threshold $time_threshold --save_name $training_data_name --n_bins $n_bins $option --seed $seed_for_dataset
./exp/naive_memories.sh 0 $dataset $data_name $training_data_name > ./exp/log/naive_memories.log &
./exp/pre_memories.sh 1 $dataset $data_name $training_data_name > ./exp/log/pre_memories.log &
./exp/tree_memories.sh 2 $dataset $data_name $training_data_name > ./exp/log/tree_memories.log & 
./exp/pre_tree_memories.sh 3 $dataset $data_name $training_data_name > ./exp/log/pre_tree_memories.log &
wait
./exp/pre_tree_single_memories.sh 0 $dataset $data_name $training_data_name > ./exp/log/pre_tree_single_memories.log &
./exp/tree_single_memories.sh 1 $dataset $data_name $training_data_name > ./exp/log/tree_single_memories.log &
./exp/laplace.sh 3 $dataset $data_name $training_data_name > ./exp/log/laplace_memories.log &