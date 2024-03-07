import os
import concurrent.futures
import pathlib

from command import command_baseline

import sys
sys.path.append("../")
from name_config import make_save_name

orig_data_dir = pathlib.Path("/data")

n_epochs = 31

n_bins = 30
dim = 32

# dataset = "peopleflow"
# max_size = 20000
# time_threshold = 30 / 60

dataset = "geolife"
max_size = 0
time_threshold = 30
epsilon = 0.0


location_threshold = 200
seeds = range(10)

save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, 0)
data_dir = orig_data_dir / dataset / str(max_size) / save_name

# conduct each command in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    
    for seed in seeds:
        command = command_baseline(data_dir, dim, seed, epsilon, n_epochs)
        combined = f"{command[0]}"
        print(combined)
        executor.submit(os.system, combined)

with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    
    for seed in seeds:
        command = command_baseline(data_dir, dim, seed, epsilon, n_epochs)
        combined = f"{command[1]}"
        print(combined)
        executor.submit(os.system, combined)