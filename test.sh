#!/usr/bin/bash

dataset=peopleflow
max_size=10000
data_name=$max_size
latlon_config=peopleflow.json
location_threshold=500
time_threshold=20
n_bins=62
seed_for_dataset=0
training_data_name=bin${n_bins}

# dataset=test
# max_size=1000
# data_name=normal_variable
# latlon_config=test.json
# location_threshold=0
# time_threshold=0
# n_bins=1
# seed_for_dataset=0
# training_data_name=seed${seed_for_dataset}_size${max_size}

# dataset=test
# max_size=1000
# data_name=circle
# latlon_config=test.json
# location_threshold=0
# time_threshold=0
# n_bins=1
# seed_for_dataset=0
# training_data_name=seed${seed_for_dataset}_size${max_size}

# dataset=test
# max_size=1000
# data_name=return
# latlon_config=test.json
# location_threshold=0
# time_threshold=0
# n_bins=1
# seed_for_dataset=0
# training_data_name=seed${seed_for_dataset}_size${max_size}

# dataset=chengdu
# max_size=100000
# data_name=${max_size}
# latlon_config=test.json
# location_threshold=0
# time_threshold=0
# n_bins=0
# seed_for_dataset=0
# training_data_name=start_end

python3 make_raw_data.py --original_data_name $dataset --max_size $max_size --seed $seed_for_dataset --save_name $data_name
python3 data_pre_processing.py --latlon_config  $latlon_config --dataset $dataset --data_name $data_name --location_threshold $location_threshold --time_threshold $time_threshold --save_name $training_data_name --n_bins $n_bins

seed=0
cuda_number=3
patience=100
batch_size=1000
noise_multiplier=1
clipping_bound=1
epsilon=6
n_split=5
n_layers=1
hidden_dim=512
embed_dim=512
meta_hidden_dim=512
learning_rate=1e-3
accountant_mode=prv
meta_network_load_path=/data/results/peopleflow/10000/bin62/test_meta_patience10000/meta_network.pt
physical_batch_size=50
n_epoch=100
meta_n_iter=100000
coef_location=1
coef_time=1
n_classes=10
global_clip=2
n_pre_epochs=0
eval_interval=10
n_test_locations=30
meta_patience=10000
privtree_theta=3000

# set the options
is_dp=True
remove_first_value=True
meta_network=True
meta_class=True
attention=True
real_start=True
post_process=False
privtree_clustering=False

declare -A arguments=(
    ["dataset"]=$dataset
    ["data_name"]=$data_name
    ["training_data_name"]=$training_data_name
    ["seed"]=$seed
    ["cuda_number"]=$cuda_number
    ["patience"]=$patience
    ["batch_size"]=$batch_size
    ["noise_multiplier"]=$noise_multiplier
    ["clipping_bound"]=$clipping_bound
    ["epsilon"]=$epsilon
    ["n_split"]=$n_split
    ["n_layers"]=$n_layers
    ["hidden_dim"]=$hidden_dim
    ["embed_dim"]=$embed_dim
    ["learning_rate"]=$learning_rate
    ["accountant_mode"]=$accountant_mode
    ["physical_batch_size"]=$physical_batch_size
    ["n_epoch"]=$n_epoch
    ["meta_n_iter"]=$meta_n_iter
    ["coef_location"]=$coef_location
    ["coef_time"]=$coef_time
    ["n_classes"]=$n_classes
    ["global_clip"]=$global_clip
    ["meta_hidden_dim"]=$meta_hidden_dim
    ["n_pre_epochs"]=$n_pre_epochs
    ["eval_interval"]=$eval_interval
    ["n_test_locations"]=$n_test_locations
    ["meta_patience"]=$meta_patience
    ["privtree_theta"]=$privtree_theta
    ["meta_network_load_path"]=$meta_network_load_path
)

declare -A options=(
    ["is_dp"]=$is_dp
    ["remove_first_value"]=$remove_first_value
    ["meta_network"]=$meta_network
    ["meta_class"]=$meta_class
    ["real_start"]=$real_start
    ["attention"]=$attention
    ["post_process"]=$post_process
    ["privtree_clustering"]=$privtree_clustering
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


epsilons=(6)
# meta_hidden_dims=(10 50 100 200)
# meta_hidden_dims=(512)
for epsilon in "${epsilons[@]}"; do
    # save_name=test_pclus${privtree_theta}_eps${epsilon}
    # save_name=test_dclus${n_classes}_eps${epsilon}
    save_name=test
    python3 run.py --save_name $save_name $option --epsilon $epsilon
done