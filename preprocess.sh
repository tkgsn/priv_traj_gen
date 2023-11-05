original_data_name=chengdu
max_size=100
seed_for_dataset=0
data_name=${max_size}

python3 make_raw_data.py --original_data_name ${original_data_name} --max_size ${max_size} --seed ${seed_for_dataset} --save_name ${data_name}

latlon_config=${original_data_name}.json
seed=0
n_bins=14
time_threshold=10
location_threshold=200

stay_data_name=${location_threshold}_${time_threshold}_bin${n_bins}_seed${seed_for_dataset}
python3 data_pre_processing.py --dataset ${original_data_name} --data_name ${data_name} --seed ${seed} --n_bins $n_bins --time_threshold $time_threshold --location_threshold $location_threshold --save_name ${stay_data_name}

route_data_name=0_0_bin${n_bins}_seed${seed_for_dataset}
python3 data_pre_processing.py --dataset ${original_data_name} --data_name ${data_name} --seed ${seed} --n_bins $n_bins --time_threshold 0 --location_threshold 0 --save_name ${route_data_name}