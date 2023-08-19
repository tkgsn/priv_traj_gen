#!/usr/bin/bash

dataset=peopleflow
max_size=100000
data_name=$max_size
latlon_config=peopleflow.json
location_threshold=500
time_threshold=20
n_bins=38
seed_for_dataset=0
training_data_name=bin38

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
cuda_number=0
patience=100
batch_size=1000
noise_multiplier=1
clipping_bound=1
epsilon=6
n_split=5
n_layers=1
hidden_dim=512
embed_dim=512
learning_rate=1e-3
accountant_mode=prv
physical_batch_size=50
n_epoch=300
meta_n_iter=1000
coef_location=1
coef_time=1
n_classes=10
global_clip=2
meta_hidden_dim=512
n_pre_epochs=0
eval_interval=10
n_test_locations=20

# set the options
is_dp=False
remove_first_value=True
fix_meta_network=False
meta_network=False
meta_class=True
attention=True
fix_embedding=True
real_start=True
post_process=False

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
)

declare -A options=(
    ["is_dp"]=$is_dp
    ["remove_first_value"]=$remove_first_value
    ["meta_network"]=$meta_network
    ["fix_meta_network"]=$fix_meta_network
    ["meta_class"]=$meta_class
    ["fix_embedding"]=$fix_embedding
    ["real_start"]=$real_start
    ["attention"]=$attention
    ["post_process"]=$post_process
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


# epsilons=(10 20 1 2 5)
seeds=(0)
for seed in "${seeds[@]}"; do
    save_name=baseline
    python3 run.py $option --save_name $save_name
done


# if the directory /data/results/test/meta_learning_variable/data_name exists, then remove it
# if [ -d "/data/results/$dataset/$data_name/$training_data_name/$save_name" ]; then
#     echo "remove /data/results/$dataset/$data_name/$training_data_name/$save_name"
#     rm -rf /data/results/$dataset/$data_name/$training_data_name/$save_name
# fi



# batch_size=100
# hidden_dim=100
# n_iter=1
# lr=1e-3

# declare -A options=(
#     ["meta_class"]=$meta_class
# )

# # make the option parameter
# option=""
# for key in "${!options[@]}"; do
#     if [ "${options[$key]}" = True ]; then
#         option="$option --$key"
#     fi
# done

# python3 run_post_training.py --cuda_number $cuda_number --dataset $dataset --data_name $data_name --training_data_name $training_data_name --batch_size $batch_size --hidden_dim $hidden_dim --n_iter $n_iter --lr $lr --save_name $save_name $option