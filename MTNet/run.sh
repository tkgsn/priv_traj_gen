#!/bin/bash

# git clone https://github.com/wangyong01/MTNet_Code

# unzip MTNet_Code
# cd MTNet_Code
# unzip MTNet.zip
# cd ..

# apt-get update
# apt-get install -y jq unzip

# data_dir=/data
# # get the data directory from "data_dir" key of config.json
# save_dir=$data_dir/chengdu/raw
# # make save_dir
# mkdir -p $save_dir

# # move the demo dataset (./MTNet_Code/MTNet/data/demo/{edge_adj.txt, edge_property.txt, trajs_demo.csv, tstamps_demo.csv}) to save_dir
# mv ./MTNet_Code/MTNet/data/demo/edge_adj.txt $save_dir
# mv ./MTNet_Code/MTNet/data/demo/edge_property.txt $save_dir
# mv ./MTNet_Code/MTNet/data/demo/trajs_demo.csv $save_dir
# mv ./MTNet_Code/MTNet/data/demo/tstamps_demo.csv $save_dir

# # remove MTNet_Code
# rm -rf MTNet_Code

pip3 install PyYaml
pip3 install geopandas

#
# VARIABLES
#
dataset=$DATASET
indice_path=$INDICE_PATH
max_size=$MAX_SIZE
dp=$DP
epoch=$EPOCH
length=$MAX_LENGTH
p_batch=$P_BATCH
#
# VARIABLES END
#

echo $indice_path

alg_name=MTNet
name=MTNET_DP${dp}

# n_bins=30
# training_data_name=200_60_bin30_seed0
# route_data_name=0_0_bin30_seed0
original_data_dir=/data/${dataset}/raw
training_data_dir=/data/${dataset}/${max_size}/${alg_name}
# route_data_dir=/data/${dataset}/${max_size}/${route_data_name}
# stay_point_data_dir=/data/${dataset}/${max_size}/${training_data_name}
save_path=/data/results/${dataset}/${max_size}/${name}
# setting_path=/data/${dataset}/${max_size}/0_0_bin${n_bins}_seed0/params.json
cuda_number=0
python3 make_training_data.py $original_data_dir $training_data_dir $dataset $indice_path
python3 train.py $training_data_dir $save_path $epoch $cuda_number $dp $length $p_batch
# python3 convert_to_original_format.py $training_data_dir $save_path
# python3 evaluate.py $save_path $route_data_dir $stay_point_data_dir $save_path