import os
import concurrent.futures
import sys
sys.path.append("../")

from name_config import make_save_name

from command import *

n_epochs = 100
epsilon = 0.0
n_bins = 30
dim = 64

dataset = "random"
max_size = 10000

time_threshold = 30
location_threshold = 200
test_thresh = 10
seeds = range(10)

data_dir = orig_data_dir / dataset / str(max_size) / make_save_name(dataset, n_bins, time_threshold, location_threshold, 0)

# conduct each command in parallel with 4 processes
with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:

    for seed in seeds:
            
        for command in [command_baseline(data_dir, dim, seed, epsilon, n_epochs, test_thresh=test_thresh), command_baseline_pre(data_dir, dim, seed, epsilon, n_epochs, test_thresh=test_thresh), command_hiemrnet(data_dir, dim, seed, epsilon, n_epochs, test_thresh=test_thresh), command_hiemrnet_pre(data_dir, dim, seed, epsilon, n_epochs, test_thresh=test_thresh), command_hiemrnet_multitask(data_dir, dim, seed, epsilon, n_epochs, test_thresh=test_thresh), command_hiemrnet_multitask_pre(data_dir, dim, seed, epsilon, n_epochs, test_thresh=test_thresh)]:
            combined = f"{command[0]}"
            print(combined)
            executor.submit(os.system, combined)


with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:

    for seed in seeds:
            
        for command in [command_baseline(data_dir, dim, seed, epsilon, n_epochs, test_thresh=test_thresh), command_baseline_pre(data_dir, dim, seed, epsilon, n_epochs, test_thresh=test_thresh), command_hiemrnet(data_dir, dim, seed, epsilon, n_epochs, test_thresh=test_thresh), command_hiemrnet_pre(data_dir, dim, seed, epsilon, n_epochs, test_thresh=test_thresh), command_hiemrnet_multitask(data_dir, dim, seed, epsilon, n_epochs, test_thresh=test_thresh), command_hiemrnet_multitask_pre(data_dir, dim, seed, epsilon, n_epochs, test_thresh=test_thresh)]:
            combined = f"{command[1]}"
            print(combined)
            executor.submit(os.system, combined)