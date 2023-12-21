#!/bin/bash

#
# VARIABLES
# 
training_data_dir=$TRAINING_DATA_DIR
seed=$SEED
meta_n_iter=$META_N_ITER
n_epoch=$EPOCH
physical_batch_size=$P_BATCH
is_dp=$DP
train_all_layers=$MULTI_TASK
consistent=$CONSISTENT
hidden_dim=$HIDDEN_DIM
location_embedding_dim=$LOC_DIM
memory_dim=$MEM_DIM
memory_hidden_dim=$MEM_HIDDEN_DIM
coef_time=$COEF_TIME
network_type=$NETWORK_TYPE
epsilon=$EPSILON
meta_dist=$META_DIST
# 
# VARIABLES END
#


data_name=$max_size
# training_data_name=${location_threshold}_${time_threshold}_bin${n_bins}_seed${seed}

# get data_directory from config.json
data_dir=$(jq -r '.data_dir' config.json)

# mkdir -p ${data_dir}/${dataset}/${data_name}
# scp -r -o StrictHostKeyChecking=no evaluation-server:${data_dir}/${dataset}/${data_name}/${training_data_name} ${data_dir}/${dataset}/${data_name}


cuda_number=0
patience=20
batch_size=0
noise_multiplier=1
clipping_bound=1
n_split=5
learning_rate=1e-3
accountant_mode=rdp
dp_delta=1e-5
meta_network_load_path=None
coef_location=1
n_classes=10
global_clip=1
meta_patience=1000
clustering=depth


# network_type=fulllinear_quadtree
# network_type=markov1

activate=relu
# activate=leaky_relu

transition_type=first
# transition_type=marginal
# transition_type=test

# set the options
remove_first_value=True
remove_duplicate=False

declare -A arguments=(
    ["training_data_dir"]=$training_data_dir
    ["seed"]=$seed
    ["cuda_number"]=$cuda_number
    ["patience"]=$patience
    ["batch_size"]=$batch_size
    ["noise_multiplier"]=$noise_multiplier
    ["clipping_bound"]=$clipping_bound
    ["epsilon"]=$epsilon
    ["n_split"]=$n_split
    ["hidden_dim"]=$hidden_dim
    ["location_embedding_dim"]=$location_embedding_dim
    ["learning_rate"]=$learning_rate
    ["accountant_mode"]=$accountant_mode
    ["physical_batch_size"]=$physical_batch_size
    ["n_epoch"]=$n_epoch
    ["meta_n_iter"]=$meta_n_iter
    ["coef_location"]=$coef_location
    ["coef_time"]=$coef_time
    ["n_classes"]=$n_classes
    ["global_clip"]=$global_clip
    ["memory_dim"]=$memory_dim
    ["meta_patience"]=$meta_patience
    ["meta_network_load_path"]=$meta_network_load_path
    ["network_type"]=$network_type
    ["activate"]=$activate
    ["meta_dist"]=$meta_dist
    ["memory_hidden_dim"]=$memory_hidden_dim
    ["clustering"]=$clustering
    ["dp_delta"]=$dp_delta
    ["transition_type"]=$transition_type
)

declare -A options=(
    ["is_dp"]=$is_dp
    ["remove_first_value"]=$remove_first_value
    ["remove_duplicate"]=$remove_duplicate
    ["train_all_layers"]=$train_all_layers
    ["consistent"]=$consistent
)

# make the option parameter
option=""
for argument in "${!arguments[@]}"; do
    option="$option --$argument ${arguments[$argument]}"
done
for key in "${!options[@]}"; do
    if [ "${options[$key]}" = True ]; then
        option="$option --$key"
    fi
done

# save_name=test
python3 run.py $option