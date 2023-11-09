#!/bin/bash

# source ./enviornment

#
# VARIABLES
# 
dataset=$DATASET
#
# VARIABLES END
#


# get the data directory from "data_dir" key of config.json
data_dir=$(jq -r '.data_dir' config.json)

# # Chengdu dataset
if [ $dataset = "chengdu" ]; then
    git clone https://github.com/wangyong01/MTNet_Code

    unzip MTNet_Code
    cd MTNet_Code
    unzip MTNet.zip
    cd ..

    save_dir=$data_dir/chengdu/raw
    # make save_dir
    mkdir -p $save_dir

    # move the demo dataset (./MTNet_Code/MTNet/data/demo/{edge_adj.txt, edge_property.txt, trajs_demo.csv, tstamps_demo.csv}) to save_dir
    mv ./MTNet_Code/MTNet/data/demo/edge_adj.txt $save_dir
    mv ./MTNet_Code/MTNet/data/demo/edge_property.txt $save_dir
    mv ./MTNet_Code/MTNet/data/demo/trajs_demo.csv $save_dir/training_data.csv
    mv ./MTNet_Code/MTNet/data/demo/tstamps_demo.csv $save_dir/training_data_time.csv

    # remove MTNet_Code
    rm -rf MTNet_Code
fi

if [ $dataset = "geolife" -o $dataset = "geolife_test" ]; then
    mkdir temp
    cd temp
    wget -O geolife_raw_data.zip -nc https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip
    unzip geolife_raw_data.zip
    save_dir=${data_dir}/$dataset
    mkdir -p $save_dir
    mv Geolife\ Trajectories\ 1.3/Data $save_dir
    cd ..
fi


python3 make_raw_data.py --dataset ${dataset}


data_dir=$(jq -r '.data_dir' config.json)
if [ $dataset = "geolife" -o $dataset = "geolife_test" ]; then
    original_data_path=${data_dir}/${dataset}/raw_data.csv
    graph_data_dir=${data_dir}/${dataset}/raw
    python3 prepare_graph.py $dataset $original_data_path $graph_data_dir
    python3 map_matching.py $graph_data_dir
    python3 make_raw_data.py --dataset ${dataset}_mm
fi