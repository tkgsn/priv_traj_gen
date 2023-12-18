import os
import concurrent.futures
import pathlib

import sys
sys.path.append("../")

from name_config import make_model_name, make_save_name

orig_data_dir = pathlib.Path("/data")


# n_binss=[6, 14, 30, 62]
n_binss = [6]
dim = 32
epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
n_epochs = 50

dataset = "geolife"
max_size = 0
time_threshold = 30
location_threshold = 200
seeds = range(1)

def command_hiemrnet(n_bins, epsilon, seed):
    meta_n_iter = 10000
    save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)
    data_dir = orig_data_dir / dataset / str(max_size) / save_name
    model_dir = data_dir / make_model_name(network_type="hiemrnet", is_dp=True, meta_n_iter=meta_n_iter, memory_dim=dim, memory_hidden_dim=dim, location_embedding_dim=dim, hidden_dim=dim, batch_size=0, train_all_layers=True, consistent=True, epsilon=epsilon, seed=seed)
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED={seed} -e META_N_ITER={meta_n_iter} -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK=True -e CONSISTENT=True -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=hiemrnet kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={model_dir} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'


with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    for seed in seeds:
        for n_bins in n_binss:
            for epsilon in epsilons:
                for command in [command_hiemrnet(n_bins, epsilon, seed)]:
                    combined = f"{command[0]}"
                    executor.submit(os.system, combined)


with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    for seed in seeds:
        for n_bins in n_binss:
            for epsilon in epsilons:
                for command in [command(n_bins, epsilon, seed)]:
                    combined = f"{command[1]}"
                    executor.submit(os.system, combined)