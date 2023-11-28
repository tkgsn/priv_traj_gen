#!/bin/bash

# apt-get install libgl1-mesa-glx
# pip install opencv-python

pip3 install pyemd

# for evaluation of MTNet
pip3 install shapely
pip3 install geopy
pip3 install cvxpy

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
ablation=$ABLATION
# 
# VARIABLES END
#

declare -A options=(
    ["ablation"]=$ablation
)

for key in "${!options[@]}"; do
    if [ "${options[$key]}" = True ]; then
        option="$option --$key"
    fi
done

echo "hello"
python3 evaluation.py --model_dir $model_dir --location_threshold $location_threshold --time_threshold $time_threshold --n_bins $n_bins --seed $seed --truncate $truncate --server $option