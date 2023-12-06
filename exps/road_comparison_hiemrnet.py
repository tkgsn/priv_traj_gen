import os
import concurrent.futures
import pathlib

import sys
sys.path.append("../")

from name_config import make_model_name, make_save_name
# from my_utils import get_data_dir
# from data_pre_processing import make_save_name
# n_binss=[6, 14, 30, 62]
# dims=[8, 16, 32, 64]

orig_data_dir = pathlib.Path("/data")

n_epochs = 31

n_bins = 30
dims = [32]

# dataset = "geolife"
# max_size = 0
# time_threshold = 30

dataset = "chengdu"
max_size = 10000

time_threshold = 30 / 60
location_threshold = 200
seeds = range(10)

# def command_baseline(n_bins, dim):
#     save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)
#     model_name = make_model_name(network_type="baseline", is_dp=True, meta_n_iter=0, memory_dim=dim, memory_hidden_dim=dim, location_embedding_dim=dim, hidden_dim=dim, batch_size=0, seed=seed)
#     data_dir = orig_data_dir / dataset / str(max_size) / save_name
#     return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK=False -e CONSISTENT=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=baseline kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
#         , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=False -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / model_name} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

# def command_pre_baseline(n_bins, dim):
#     meta_n_iter = 10000
#     save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)
#     data_dir = orig_data_dir / dataset / str(max_size) / save_name
#     return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER={meta_n_iter} -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK=False -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=baseline kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
#     , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / f"baseline_dpTrue_meta{meta_n_iter}_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_trFalse_coFalse_mulFalse_test"} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

# # deconv
# def command_hiemrnet(n_bins, dim):
#     save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)
#     data_dir = orig_data_dir / dataset / str(max_size) / save_name
#     return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER=0 -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK=False -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=hiemrnet kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
#         , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / f"hiemrnet_dpTrue_meta0_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_trFalse_coFalse_mulFalse_test"} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

# # deconv + multi task
# def command_hiemrnet_multitask(n_bins, dim):
#     save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)
#     data_dir = orig_data_dir / dataset / str(max_size) / save_name
#     multi_task = "True"
#     return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER=0 -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK={multi_task} -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=hiemrnet kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
#         , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / f"hiemrnet_dpTrue_meta0_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_tr{multi_task}_coFalse_mulFalse_test"} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

# # deconv + pre-trainig
# def command_pre_hiemrnet(n_bins, dim):
#     save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)
#     data_dir = orig_data_dir / dataset / str(max_size) / save_name
#     meta_n_iter = 10000
#     return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER={meta_n_iter} -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK=False -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=hiemrnet kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
#         , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / f"hiemrnet_dpTrue_meta{meta_n_iter}_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_trFalse_coFalse_mulFalse_test"} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

# # deconv + pre-trainig + multi task
# def command_hiemrnet_pre_multitask(n_bins, dim):
#     save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, seed)
#     data_dir = orig_data_dir / dataset / str(max_size) / save_name
#     multi_task = "True"
#     meta_n_iter = 10000
#     return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER={meta_n_iter} -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK={multi_task} -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=hiemrnet kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
#         , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / f"hiemrnet_dpTrue_meta{meta_n_iter}_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_tr{multi_task}_coFalse_mulFalse_test"} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

# deconv + pre-trainig + multi task + consistent
def command_hiemrnet_pre_multitask_consistent(n_bins, dim, seed):
    save_name = make_save_name(dataset, n_bins, time_threshold, location_threshold, 0)
    data_dir = orig_data_dir / dataset / str(max_size) / save_name
    multi_task = "True"
    meta_n_iter = 10000
    consistent = "True"
    model_name = make_model_name(network_type="hiemrnet", is_dp=True, meta_n_iter=meta_n_iter, memory_dim=dim, memory_hidden_dim=dim, location_embedding_dim=dim, hidden_dim=dim, batch_size=0, train_all_layers=multi_task, consistent=consistent, seed=seed)
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED={seed} -e META_N_ITER={meta_n_iter} -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK={multi_task} -e CONSISTENT={consistent} -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=hiemrnet kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=False -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={data_dir / model_name} -e EVAL_INTERVAL=1 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'


# conduct each command in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        
    for dim in dims:
        for seed in seeds:
            command = command_hiemrnet_pre_multitask_consistent(n_bins, dim, seed)
            combined = f"{command[0]}"
            print(combined)
            executor.submit(os.system, combined)


with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        
    for dim in dims:
        for seed in seeds:
            command = command_hiemrnet_pre_multitask_consistent(n_bins, dim, seed)
            combined = f"{command[1]}"
            print(combined)
            executor.submit(os.system, combined)