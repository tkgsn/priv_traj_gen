import os
import concurrent.futures
import pathlib
from name_config import make_model_name, make_save_name
import sys
sys.path.append("../")

import json
from command import *

with open("../config.json") as f:
    config = json.load(f)

orig_data_dir = config["data_dir"]

n_epochs = 100
epsilon = 0

n_bins = 14
dim = 64

dataset = "geolife"
max_size = 0
time_threshold = 30
location_threshold = 200
seeds = [0]
seeds = range(1)

data_dir = orig_data_dir / dataset / str(max_size) / make_save_name(dataset, n_bins, time_threshold, location_threshold, 0)


# conduct each command in parallel with 4 processes
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    
    for seed in seeds:
        for command in [\
                command_baseline(data_dir, dim, seed, epsilon, n_epochs),\
                command_baseline_pre(data_dir, dim, seed, epsilon, n_epochs),\
                command_hiemrnet(data_dir, dim, seed, epsilon, n_epochs),\
                command_hiemrnet_multitask(data_dir, dim, seed, epsilon, n_epochs),\
                command_hiemrnet_pre(data_dir, dim, seed, epsilon, n_epochs), \
                command_hiemrnet_multitask_pre(data_dir, dim, seed, epsilon, n_epochs), \
                ]:
            combined = f"{command[0]}; {command[1]}"
            print(combined)
            executor.submit(os.system, combined)