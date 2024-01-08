from name_config import make_model_name, make_save_name
import json
import pathlib
import sys
sys.path.append("../")

with open("../config.json") as f:
    config = json.load(f)

orig_data_dir = pathlib.Path(config["data_dir"])



def command_baseline(data_dir, dim, seed, epsilon, n_epochs, test_thresh=30):
    meta_n_iter = 0
    network_type = "baseline"
    meta_dist = "both"
    transition_type = "first"
    train_all_layers = False
    model_dir = data_dir / make_model_name(network_type=network_type, is_dp=True, meta_n_iter=meta_n_iter, memory_dim=dim, memory_hidden_dim=dim, location_embedding_dim=dim, hidden_dim=dim, batch_size=0, train_all_layers=train_all_layers, consistent=True, epsilon=epsilon, seed=seed, meta_dist=meta_dist, transition_type=transition_type)
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED={seed} -e META_N_ITER={meta_n_iter} -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK={train_all_layers} -e CONSISTENT=True -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e EPSILON={epsilon} -e META_DIST={meta_dist} -e COEF_TIME=1 -e NETWORK_TYPE={network_type} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH={test_thresh} -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={model_dir} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'


# p
def command_baseline_pre(data_dir, dim, seed, epsilon, n_epochs, test_thresh=30):
    meta_n_iter = 10000
    network_type = "baseline"
    meta_dist = "both"
    transition_type = "first"
    train_all_layers = False
    model_dir = data_dir / make_model_name(network_type=network_type, is_dp=True, meta_n_iter=meta_n_iter, memory_dim=dim, memory_hidden_dim=dim, location_embedding_dim=dim, hidden_dim=dim, batch_size=0, train_all_layers=train_all_layers, consistent=True, epsilon=epsilon, seed=seed, meta_dist=meta_dist, transition_type=transition_type)
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED={seed} -e META_N_ITER={meta_n_iter} -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK={train_all_layers} -e CONSISTENT=True -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e EPSILON={epsilon} -e META_DIST={meta_dist} -e COEF_TIME=1 -e NETWORK_TYPE={network_type} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH={test_thresh} -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={model_dir} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'


# d
def command_hiemrnet(data_dir, dim, seed, epsilon, n_epochs, test_thresh=30):
    meta_n_iter = 0
    network_type = "hiemrnet"
    meta_dist = "both"
    transition_type = "first"
    train_all_layers = False
    model_dir = data_dir / make_model_name(network_type=network_type, is_dp=True, meta_n_iter=meta_n_iter, memory_dim=dim, memory_hidden_dim=dim, location_embedding_dim=dim, hidden_dim=dim, batch_size=0, train_all_layers=train_all_layers, consistent=True, epsilon=epsilon, seed=seed, meta_dist=meta_dist, transition_type=transition_type)
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED={seed} -e META_N_ITER={meta_n_iter} -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK={train_all_layers} -e CONSISTENT=True -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e EPSILON={epsilon} -e META_DIST={meta_dist} -e COEF_TIME=1 -e NETWORK_TYPE={network_type} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH={test_thresh} -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={model_dir} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'

# d+p
def command_hiemrnet_pre(data_dir, dim, seed, epsilon, n_epochs, test_thresh=30):
    meta_n_iter = 10000
    network_type = "hiemrnet"
    meta_dist = "both"
    transition_type = "first"
    train_all_layers = False
    model_dir = data_dir / make_model_name(network_type=network_type, is_dp=True, meta_n_iter=meta_n_iter, memory_dim=dim, memory_hidden_dim=dim, location_embedding_dim=dim, hidden_dim=dim, batch_size=0, train_all_layers=train_all_layers, consistent=True, epsilon=epsilon, seed=seed, meta_dist=meta_dist, transition_type=transition_type)
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED={seed} -e META_N_ITER={meta_n_iter} -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK={train_all_layers} -e CONSISTENT=True -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e EPSILON={epsilon} -e META_DIST={meta_dist} -e COEF_TIME=1 -e NETWORK_TYPE={network_type} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH={test_thresh} -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={model_dir} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'



# d+m
def command_hiemrnet_multitask(data_dir, dim, seed, epsilon, n_epochs, test_thresh=30):
    meta_n_iter = 0
    network_type = "hiemrnet"
    meta_dist = "both"
    transition_type = "first"
    train_all_layers = True
    model_dir = data_dir / make_model_name(network_type=network_type, is_dp=True, meta_n_iter=meta_n_iter, memory_dim=dim, memory_hidden_dim=dim, location_embedding_dim=dim, hidden_dim=dim, batch_size=0, train_all_layers=train_all_layers, consistent=True, epsilon=epsilon, seed=seed, meta_dist=meta_dist, transition_type=transition_type)
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED={seed} -e META_N_ITER={meta_n_iter} -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK={train_all_layers} -e CONSISTENT=True -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e EPSILON={epsilon} -e META_DIST={meta_dist} -e COEF_TIME=1 -e NETWORK_TYPE={network_type} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH={test_thresh} -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={model_dir} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'


# d+m+p
def command_hiemrnet_multitask_pre(data_dir, dim, seed, epsilon, n_epochs, test_thresh=30):
    meta_n_iter = 10000
    network_type = "hiemrnet"
    meta_dist = "both"
    transition_type = "first"
    train_all_layers = True
    model_dir = data_dir / make_model_name(network_type=network_type, is_dp=True, meta_n_iter=meta_n_iter, memory_dim=dim, memory_hidden_dim=dim, location_embedding_dim=dim, hidden_dim=dim, batch_size=0, train_all_layers=train_all_layers, consistent=True, epsilon=epsilon, seed=seed, meta_dist=meta_dist, transition_type=transition_type)
    return f'docker run --rm --gpus all -v /mnt/data:/data -e TRAINING_DATA_DIR={data_dir} -e SEED={seed} -e META_N_ITER={meta_n_iter} -e EPOCH={n_epochs} -e P_BATCH=100 -e DP=True -e MULTI_TASK={train_all_layers} -e CONSISTENT=True -e HIDDEN_DIM={dim} -e LOC_DIM={dim} -e MEM_DIM={dim} -e MEM_HIDDEN_DIM={dim} -e EPSILON={epsilon} -e META_DIST={meta_dist} -e COEF_TIME=1 -e NETWORK_TYPE={network_type} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./train.sh"' \
        , f'docker run --rm --gpus all -v /mnt/data:/data -e TEST_THRESH={test_thresh} -e ABLATION=True -e SEED=0 -e TRUNCATE=0 -e MODEL_DIR={model_dir} -e EVAL_INTERVAL=10 -e EVAL_DATA_DIR={data_dir} kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./evaluate.sh"'