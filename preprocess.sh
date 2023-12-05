#!/bin/bash

# source ./enviornment

#
# VARIABLES
#
# seed=0
# max_size=0
# n_bins=30
# time_threshold=10
# location_threshold=200
dataset=$DATASET
seed=$SEED
max_size=$MAX_SIZE
n_bins=$N_BINS
time_threshold=$T_THRESH
location_threshold=$L_THRESH
truncate=$TRUNCATE
#
# VARIABLES END
#

# get the data directory from "data_dir" key of config.json
data_dir=$(jq -r '.data_dir' config.json)

data_name=${max_size}

stay_data_name=${location_threshold}_${time_threshold}_bin${n_bins}_seed${seed}
python3 data_pre_processing.py --dataset ${dataset} --data_name ${data_name} --max_size ${max_size} --seed ${seed} --n_bins $n_bins --time_threshold $time_threshold --location_threshold $location_threshold --save_name ${stay_data_name} --truncate $truncate

# route_data_name=0_0_bin${n_bins}_seed${seed}
# python3 data_pre_processing.py --dataset ${dataset} --data_name ${data_name} --max_size ${max_size} --seed ${seed} --n_bins $n_bins --time_threshold 0 --location_threshold 0 --save_name ${route_data_name} --truncate $truncate