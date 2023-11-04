#!/usr/bin/bash

git clone https://github.com/wangyong01/MTNet_Code

unzip MTNet_Code
cd MTNet_Code
unzip MTNet.zip
cd ..

apt-get update
apt-get install jq

# get the data directory from "data_dir" key of config.json
data_dir=$(jq -r '.data_dir' config.json)
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