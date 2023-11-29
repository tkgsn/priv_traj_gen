import os
import concurrent.futures

n_binss=[6, 14, 30, 62]
dims=[8, 16, 32, 64]
n_epochs = 100

def command_baseline(n_bins, dim):
    return f'docker run -it --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR=/data/rotation/10000/bin{n_bins}_seed0 -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER=0 -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK=False -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=baseline kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run -it --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR=/data/rotation/10000/bin{n_bins}_seed0/baseline_dpTrue_meta0_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_trFalse_coFalse_mulFalse_test -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR=/data/rotation/10000/bin{n_bins}_seed0 kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"', \

def command_pre_baseline(n_bins, dim):
    meta_n_iter = 10000
    return f'docker run -it --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR=/data/rotation/10000/bin{n_bins}_seed0 -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER={meta_n_iter} -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK=False -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=baseline kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
    , f'docker run -it --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR=/data/rotation/10000/bin{n_bins}_seed0/baseline_dpTrue_meta{meta_n_iter}_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_trFalse_coFalse_mulFalse_test -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR=/data/rotation/10000/bin{n_bins}_seed0 kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

def command_hiemrnet(n_bins, dim):
    return f'docker run -it --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR=/data/rotation/10000/bin{n_bins}_seed0 -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER=0 -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK=False -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=hiemrnet kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run -it --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR=/data/rotation/10000/bin{n_bins}_seed0/hiemrnet_dpTrue_meta0_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_trFalse_coFalse_mulFalse_test -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR=/data/rotation/10000/bin{n_bins}_seed0 kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

def command_pre_hiemrnet(n_bins, dim):
    meta_n_iter = 10000
    return f'docker run -it --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR=/data/rotation/10000/bin{n_bins}_seed0 -e SEED=0 -e TRAINING_SEED=0 -e META_N_ITER={meta_n_iter} -e SEED=0 -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK=False -e CONSISTENT=False -e MULTILAYER=False -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e COEF_TIME=1 -e NETWORK_TYPE=hiemrnet kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run -it --gpus all -v /mnt/data:/data -e TEST_THRESH=30 -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR=/data/rotation/10000/bin{n_bins}_seed0/hiemrnet_dpTrue_meta{meta_n_iter}_dim{dim}_{dim}_{dim}_{dim}_btch0_cldepth_1000_trFalse_coFalse_mulFalse_test -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR=/data/rotation/10000/bin{n_bins}_seed0 kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

# conduct each command in parallel with 4 processes
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    for n_bins in n_binss:
        for dim in dims:
            for command in [command_baseline(n_bins, dim), command_pre_baseline(n_bins, dim), command_hiemrnet(n_bins, dim), command_pre_hiemrnet(n_bins, dim)]:
                combined = f"{command[0]} && {command[1]}"
                executor.submit(os.system, combined)