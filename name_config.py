import json
import os
from my_utils import get_datadir

def result_name(i, consistent):
    return f"evaluated_{i}_co.json" if consistent else f"evaluated_{i}.json"

def make_raw_data_path(dataset, **kwargs):
    return get_datadir() / dataset / "raw_data.csv"

def make_training_data_path(dataset_name, size, save_name, **kwargs):
    return get_datadir() / dataset_name / f"{size}" / save_name

def make_save_name(dataset_name, n_bins, time_threshold, location_threshold, dataset_seed, **kwargs):

    if dataset_name in ["rotation", "random", "chengdu", "geolife_mm"]:
        save_name = f"bin{n_bins}_seed{dataset_seed}"
    elif dataset_name == "peopleflow":
        save_name = f"{location_threshold}_{int(time_threshold*60)}_bin{n_bins}_seed{dataset_seed}"
    else:
        save_name = f"{location_threshold}_{time_threshold}_bin{n_bins}_seed{dataset_seed}"
    return save_name

def make_model_name(**kwargs):
    if kwargs["model_name"] == "hrnet":
        if kwargs["multitask"] == False:
            kwargs["consistent"] = False
        # save_name = f"{kwargs['model_name']}_dp{kwargs['is_dp']}_{kwargs['meta_dist']}{kwargs['pre_n_iter']}_dim{kwargs['memory_dim']}_{kwargs['memory_hidden_dim']}_{kwargs['location_embedding_dim']}_{kwargs['hidden_dim']}_btch{kwargs['batch_size']}_tr{kwargs['multitask']}_co{kwargs['consistent']}_eps{kwargs['epsilon']}_{kwargs['model_seed']}"
        # save_name = f"{kwargs['model_name']}_dp{kwargs['is_dp']}_{kwargs['meta_dist']}{kwargs['pre_n_iter']}_dim{kwargs['memory_dim']}_{kwargs['memory_hidden_dim']}_{kwargs['location_embedding_dim']}_btch{kwargs['batch_size']}_mul{kwargs['multitask']}_eps{kwargs['epsilon']}_{kwargs['model_seed']}"
        save_name = f"{kwargs['model_name']}_dp{kwargs['is_dp']}_{kwargs['meta_dist']}{kwargs['pre_n_iter']}_{kwargs['memory_hidden_dim']}_{kwargs['location_embedding_dim']}_btch{kwargs['batch_size']}_mul{kwargs['multitask']}_eps{kwargs['epsilon']}_{kwargs['model_seed']}"
    elif kwargs["model_name"] == "baseline":
        save_name = f"{kwargs['model_name']}_dp{kwargs['is_dp']}_pre{kwargs['pre_n_iter']}_dim{kwargs['memory_hidden_dim']}_{kwargs['location_embedding_dim']}_btch{kwargs['batch_size']}_eps{kwargs['epsilon']}_{kwargs['model_seed']}"
    elif kwargs["model_name"] == "mtnet":
        save_name = f"{kwargs['model_name']}_dp{kwargs['is_dp']}_{kwargs['model_seed']}"

    if kwargs["transition_type"] == "test":
        save_name = f"test_eps{kwargs['epsilon']}"
    return save_name

def make_model_dir(**kwargs):
    save_name = make_save_name(**kwargs)
    return make_training_data_path(save_name=save_name, **kwargs) / make_model_name(**kwargs)

directory = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(directory, "config.json"), "r") as f:
    config = json.load(f)
image_name = config["image_name"]