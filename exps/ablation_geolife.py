import os
import concurrent.futures
import pathlib
# from my_utils import get_data_dir
# from data_pre_processing import make_save_name
# n_binss=[6, 14, 30, 62]
# dims=[8, 16, 32, 64]

orig_data_dir = pathlib.Path("/data")

def make_save_name(dataset_name, n_bins, time_threshold, location_threshold, seed):

    if dataset_name in ["rotation", "random"]:
        save_name = f"bin{n_bins}_seed{seed}"
    else:
        save_name = f"{location_threshold}_{time_threshold}_bin{n_bins}_seed{seed}"
    return save_name



n_epochs = 100

n_bins = 30
dim = 64

dataset = "geolife_mm"
max_size = 0

time_threshold = 30
location_threshold = 200
seed = 0

def command_baseline(n_bins, dim):
    save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)
    data_dir = orig_data_dir / dataset / str(max_size) / save_name
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER=0 -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK=False -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=baseline kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / f"baseline_dpTrue_meta0_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_trFalse_coFalse_mulFalse_test"} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"', \

def command_pre_baseline(n_bins, dim):
    meta_n_iter = 10000
    save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)
    data_dir = orig_data_dir / dataset / str(max_size) / save_name
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER={meta_n_iter} -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK=False -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=baseline kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
    , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / f"baseline_dpTrue_meta{meta_n_iter}_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_trFalse_coFalse_mulFalse_test"} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

# deconv
def command_hiemrnet(n_bins, dim):
    save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)
    data_dir = orig_data_dir / dataset / str(max_size) / save_name
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER=0 -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK=False -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=hiemrnet kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / f"hiemrnet_dpTrue_meta0_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_trFalse_coFalse_mulFalse_test"} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

# deconv + multi task
def command_hiemrnet_multitask(n_bins, dim):
    save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)
    data_dir = orig_data_dir / dataset / str(max_size) / save_name
    multi_task = "True"
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER=0 -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK={multi_task} -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=hiemrnet kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / f"hiemrnet_dpTrue_meta0_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_tr{multi_task}_coFalse_mulFalse_test"} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

# deconv + pre-trainig
def command_pre_hiemrnet(n_bins, dim):
    save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)
    data_dir = orig_data_dir / dataset / str(max_size) / save_name
    meta_n_iter = 10000
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER={meta_n_iter} -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK=False -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=hiemrnet kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / f"hiemrnet_dpTrue_meta{meta_n_iter}_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_trFalse_coFalse_mulFalse_test"} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

# deconv + pre-trainig + multi task
def command_hiemrnet_pre_multitask(n_bins, dim):
    save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)
    data_dir = orig_data_dir / dataset / str(max_size) / save_name
    multi_task = "True"
    meta_n_iter = 10000
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER={meta_n_iter} -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK={multi_task} -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=hiemrnet kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / f"hiemrnet_dpTrue_meta{meta_n_iter}_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_tr{multi_task}_coFalse_mulFalse_test"} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'


# conduct each command in parallel with 4 processes
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:

    for command in [command_baseline(n_bins, dim), command_pre_baseline(n_bins, dim), command_hiemrnet(n_bins, dim), command_pre_hiemrnet(n_bins, dim)]:
        combined = f"{command[0]}; {command[1]}"
        print(combined)
        executor.submit(os.system, combined)