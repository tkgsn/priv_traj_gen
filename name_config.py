
def make_save_name(dataset_name, n_bins, time_threshold, location_threshold, seed):

    if dataset_name in ["rotation", "random", "chengdu", "geolife_mm"]:
        save_name = f"bin{n_bins}_seed{seed}"
    elif dataset_name == "peopleflow":
        save_name = f"{location_threshold}_{int(time_threshold*60)}_bin{n_bins}_seed{seed}"
    else:
        save_name = f"{location_threshold}_{time_threshold}_bin{n_bins}_seed{seed}"
    return save_name

def make_model_name(**kwargs):
    if kwargs["network_type"] == "hiemrnet":
        save_name = f"{kwargs['network_type']}_dp{kwargs['is_dp']}_pre{kwargs['meta_n_iter']}_dim{kwargs['memory_dim']}_{kwargs['memory_hidden_dim']}_{kwargs['location_embedding_dim']}_{kwargs['hidden_dim']}_btch{kwargs['batch_size']}_tr{kwargs['train_all_layers']}_co{kwargs['consistent']}_eps{kwargs['epsilon']}_{kwargs['seed']}"
    elif kwargs["network_type"] == "baseline":
        save_name = f"{kwargs['network_type']}_dp{kwargs['is_dp']}_pre{kwargs['meta_n_iter']}_dim{kwargs['memory_dim']}_{kwargs['memory_hidden_dim']}_{kwargs['location_embedding_dim']}_{kwargs['hidden_dim']}_btch{kwargs['batch_size']}_{kwargs['seed']}"
    elif kwargs["network_type"] == "mtnet":
        save_name = f"{kwargs['network_type']}_dp{kwargs['is_dp']}_{kwargs['seed']}"
    return save_name