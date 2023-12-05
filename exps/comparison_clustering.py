import os
import concurrent.futures
import pathlib

orig_data_dir = pathlib.Path("/data")

dataset = "geolife"
n_bins = 30
max_size = 0

time_threshold = 30
location_threshold = 200
seed = 0
epsilons = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
# epsilons = [0.0]
ks = [300]
# epsilons = [2.0]
seeds = range(10)

def make_save_name(dataset_name, n_bins, time_threshold, location_threshold, seed):

    if dataset_name in ["rotation", "random"]:
        save_name = f"bin{n_bins}_seed{seed}"
    else:
        save_name = f"{location_threshold}_{time_threshold}_bin{n_bins}_seed{seed}"
    return save_name


data_dir = orig_data_dir / dataset / str(max_size) / make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)

def command(epsilon, k, seed):
    return f'docker run --rm -v /mnt/data:/data -e SEED={seed} -e DATASET={dataset} -e MAX_SIZE={max_size} -e EPSILON={epsilon} -e K={k} -e N_BINS={n_bins} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "cd ./competitors/clustering && ./run.sh"' ,\
            f'docker run --rm -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=False -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / f"clustering_{k}_{seed}" / f"model_{epsilon}"} -e EVAL_INTERVAL=1 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    for k in ks:
        for epsilon in epsilons:
            for seed in seeds:
                command1, command2 = command(epsilon, k, seed)
                combined = f"{command1}; {command2}"
                executor.submit(os.system, combined)