#!/usr/bin/bash


# dataset=peopleflow
# max_size=10000
# data_name=$max_size
# latlon_config=peopleflow.json
# location_threshold=500
# time_threshold=20
# n_bins=30
# seed_for_dataset=0
# training_data_name=bin${n_bins}_seed${seed_for_dataset}


# dataset=test
# max_size=1000
# data_name=normal_variable
# latlon_config=test.json
# location_threshold=0
# time_threshold=0
# n_bins=2
# seed_for_dataset=0
# training_data_name=seed${seed_for_dataset}_size${max_size}_nbins${n_bins}

# dataset=rotation
# max_size=1000
# data_name=$max_size
# latlon_config=test.json
# location_threshold=0
# time_threshold=0
# n_bins=30
# seed_for_dataset=0
# training_data_name=bin${n_bins}_seed${seed_for_dataset}

# dataset=rotation
# max_size=10000
# data_name=$max_size
# latlon_config=test.json
# location_threshold=0
# time_threshold=0
# n_bins=62
# seed_for_dataset=0
# training_data_name=bin${n_bins}_seed${seed_for_dataset}

# dataset=random
# max_size=100000
# data_name=$max_size
# latlon_config=test.json
# location_threshold=0
# time_threshold=0
# n_bins=30
# seed_for_dataset=0
# training_data_name=bin${n_bins}_seed${seed_for_dataset}

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

# dataset=test
# max_size=1000
# data_name=quadtree
# latlon_config=test.json
# location_threshold=0
# time_threshold=0
# n_bins=6
# seed_for_dataset=0
# training_data_name=seed${seed_for_dataset}_size${max_size}_nbins${n_bins}


# dataset=chengdu
# max_size=10000
# data_name=${max_size}
# latlon_config=chengdu.json
# location_threshold=200
# time_threshold=60
# n_bins=30
# seed_for_dataset=0


# option=""
# dataset=chengdu
# max_size=10000
# data_name=${max_size}
# latlon_config=chengdu.json
# location_threshold=0
# time_threshold=0
# n_bins=30
# seed_for_dataset=0


# dataset=geolife
# max_size=0
# data_name=${max_size}
# latlon_config=geolife.json
# location_threshold=200
# time_threshold=10
# n_bins=30
# seed_for_dataset=0

dataset=geolife_mm
max_size=0
data_name=${max_size}
latlon_config=geolife.json
location_threshold=200
time_threshold=10
n_bins=30
seed_for_dataset=0


# dataset=taxi
# max_size=10000
# data_name=${max_size}
# latlon_config=taxi.json
# location_threshold=500
# time_threshold=20
# n_bins=62
# seed_for_dataset=0
# startend=True
# training_data_name=seed${seed_for_dataset}_bin${n_bins}_startend${startend}
# if [ ${startend} = True ]; then
#     option=--startend
# fi

training_data_name=${location_threshold}_${time_threshold}_bin${n_bins}_seed${seed_for_dataset}
python3 make_raw_data.py --original_data_name $dataset --max_size $max_size --seed $seed_for_dataset --save_name $data_name --n_bins $n_bins
python3 data_pre_processing.py --latlon_config  $latlon_config --dataset $dataset --data_name $data_name --location_threshold $location_threshold --time_threshold $time_threshold --save_name $training_data_name --n_bins $n_bins --seed $seed_for_dataset

route_data_name=0_0_bin${n_bins}_seed${seed_for_dataset}
python3 data_pre_processing.py --latlon_config  $latlon_config --dataset $dataset --data_name $data_name --location_threshold 0 --time_threshold 0 --save_name $route_data_name --n_bins $n_bins --seed $seed_for_dataset

#!/usr/bin/bash

# cuda_number=3
# seed=0
# patience=10
# batch_size=0
# noise_multiplier=1
# clipping_bound=1
# epsilon=1000
# n_split=5
# n_layers=1
# hidden_dim=256
# location_embedding_dim=64
# memory_dim=100
# memory_hidden_dim=64
# learning_rate=1e-3
# accountant_mode=rdp
# dp_delta=1e-5
# meta_network_load_path=None
# physical_batch_size=20
# n_epoch=1000
# meta_n_iter=10
# coef_location=1
# coef_time=1
# n_classes=10
# global_clip=1
# n_pre_epochs=0
# eval_interval=1
# n_test_locations=30
# meta_patience=1000
# meta_dist=dirichlet
# # meta_dist=eye
# privtree_theta=3000
# clustering=depth
# # clustering=privtree
# # clustering=distance

# # network_type=meta_network
# network_type=fulllinear_quadtree
# # network_type=markov1

# activate=relu
# # activate=leaky_relu

# transition_type=first
# # transition_type=marginal

# # set the options
# is_dp=True
# remove_first_value=True
# remove_duplicate=False
# real_start=True
# post_process=False
# eval_initial=True
# train_all_layers=True
# consistent=True
# compensation=True


# evaluate_second_next_location=False
# evaluate_second_order_next_location=False
# evaluate_first_next_location=False
# evaluate_global=True
# evaluate_source=True
# evaluate_target=True
# evaluate_route=True
# evaluate_destination=True
# evaluate_distance=True
# evaluate_passing=True
# evaluate_empirical_next_location=False

# declare -A arguments=(
#     ["dataset"]=$dataset
#     ["data_name"]=$data_name
#     ["training_data_name"]=$training_data_name
#     ["seed"]=$seed
#     ["cuda_number"]=$cuda_number
#     ["patience"]=$patience
#     ["batch_size"]=$batch_size
#     ["noise_multiplier"]=$noise_multiplier
#     ["clipping_bound"]=$clipping_bound
#     ["epsilon"]=$epsilon
#     ["n_split"]=$n_split
#     ["n_layers"]=$n_layers
#     ["hidden_dim"]=$hidden_dim
#     ["location_embedding_dim"]=$location_embedding_dim
#     ["learning_rate"]=$learning_rate
#     ["accountant_mode"]=$accountant_mode
#     ["physical_batch_size"]=$physical_batch_size
#     ["n_epoch"]=$n_epoch
#     ["meta_n_iter"]=$meta_n_iter
#     ["coef_location"]=$coef_location
#     ["coef_time"]=$coef_time
#     ["n_classes"]=$n_classes
#     ["global_clip"]=$global_clip
#     ["memory_dim"]=$memory_dim
#     ["n_pre_epochs"]=$n_pre_epochs
#     ["eval_interval"]=$eval_interval
#     ["n_test_locations"]=$n_test_locations
#     ["meta_patience"]=$meta_patience
#     ["privtree_theta"]=$privtree_theta
#     ["meta_network_load_path"]=$meta_network_load_path
#     ["network_type"]=$network_type
#     ["activate"]=$activate
#     ["meta_dist"]=$meta_dist
#     ["memory_hidden_dim"]=$memory_hidden_dim
#     ["clustering"]=$clustering
#     ["dp_delta"]=$dp_delta
#     ["transition_type"]=$transition_type
#     ["route_data_name"]=$route_data_name
# )

# declare -A options=(
#     ["is_dp"]=$is_dp
#     ["remove_first_value"]=$remove_first_value
#     ["remove_duplicate"]=$remove_duplicate
#     ["real_start"]=$real_start
#     ["post_process"]=$post_process
#     ["eval_initial"]=$eval_initial
#     ["train_all_layers"]=$train_all_layers
#     ["evaluate_first_next_location"]=$evaluate_first_next_location
#     ["evaluate_second_next_location"]=$evaluate_second_next_location
#     ["evaluate_second_order_next_location"]=$evaluate_second_order_next_location
#     ["evaluate_global"]=$evaluate_global
#     ["evaluate_source"]=$evaluate_source
#     ["evaluate_target"]=$evaluate_target
#     ["evaluate_route"]=$evaluate_route
#     ["evaluate_destination"]=$evaluate_destination
#     ["evaluate_distance"]=$evaluate_distance
#     ["evaluate_empirical_next_location"]=$evaluate_empirical_next_location
#     ["evaluate_passing"]=$evaluate_passing
#     ["consistent"]=$consistent
#     ["compensation"]=$compensation
# )

# # make the option parameter
# option=""
# for argument in "${!arguments[@]}"; do
#     option="$option --$argument ${arguments[$argument]}"
# done
# for key in "${!options[@]}"; do
#     if [ "${options[$key]}" = True ]; then
#         option="$option --$key"
#     fi
# done


# save_name=${network_type}_dp${is_dp}_meta${meta_n_iter}_dim${memory_dim}_${memory_hidden_dim}_${location_embedding_dim}_${hidden_dim}_btch${batch_size}_cl${clustering}_${epsilon}_tr${train_all_layers}_co${consistent}
# # save_name=test
# python3 run.py --save_name $save_name $option