#!/bin/bash

# apt-get update
# apt-get install -y jq unzip
# # apt install -y python2
# # curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py
# # python2 get-pip.py

# apt-get install -y software-properties-common
# add-apt-repository -y ppa:ubuntugis/ppa
# apt-get -q update
# apt-get -y install libboost-dev libboost-serialization-dev gdal-bin libgdal-dev make cmake libbz2-dev libexpat1-dev swig python-dev build-essential

# git clone https://github.com/cyang-kth/fmm.git
# cd fmm

# mkdir build
# cd build
# cmake ..
# make -j4
# make install

# # get the data directory from "data_dir" key of config.json
data_dir=$(jq -r '.data_dir' config.json)

dataset=geolife_test_mm
seed=0
max_size=0
n_bins=30
time_threshold=10
location_threshold=200
data_name=${max_size}

stay_data_name=${location_threshold}_${time_threshold}_bin${n_bins}_seed${seed}
python3 data_pre_processing.py --dataset ${dataset} --data_name ${data_name} --max_size ${max_size} --seed ${seed} --n_bins $n_bins --time_threshold $time_threshold --location_threshold $location_threshold --save_name ${stay_data_name}

route_data_name=0_0_bin${n_bins}_seed${seed}
python3 data_pre_processing.py --dataset ${dataset} --data_name ${data_name} --max_size ${max_size} --seed ${seed} --n_bins $n_bins --time_threshold 0 --location_threshold 0 --save_name ${route_data_name}