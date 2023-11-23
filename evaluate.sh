#!/bin/bash

# apt-get install libgl1-mesa-glx
# pip install opencv-python

pip install pyemd

# source ./enviornment
#
# VARIABLE
# 
model_dir=$MODEL_DIR
location_threshold=$L_THRESH
time_threshold=$T_THRESH
n_bins=$N_BINS
seed=$SEED
truncate=$TRUNCATE
# 
# VARIABLES END
#

echo "hello"
python3 evaluation.py --model_dir $model_dir --location_threshold $location_threshold --time_threshold $time_threshold --n_bins $n_bins --seed $seed --truncate $truncate --server