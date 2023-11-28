#!/bin/bash

# apt-get install libgl1-mesa-glx
# pip install opencv-python

pip3 install pyemd shapely geopy cvxpy

# for evaluation of MTNet

# source ./enviornment
#
# VARIABLE
# 
model_dir=$MODEL_DIR
# location_threshold=$L_THRESH
# time_threshold=$T_THRESH
# n_bins=$EVALUATE_N_BINS
seed=$SEED
truncate=$TRUNCATE
ablation=$ABLATION
eval_data_dir=$EVAL_DATA_DIR
test_thresh=$TEST_THRESH
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
python3 evaluation.py --model_dir $model_dir --eval_data_dir $eval_data_dir --seed $seed --truncate $truncate --test_thresh $test_thresh $option