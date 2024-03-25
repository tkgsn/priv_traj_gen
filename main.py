import argparse
import random
import numpy as np
import torch
import tqdm
from torch import nn, optim
import json
from scipy.spatial.distance import jensenshannon
from collections import Counter
import pathlib

from name_config import make_model_name, make_save_name, make_raw_data_path, make_training_data_path
from my_utils import get_datadir, privtree_clustering, depth_clustering, noise_normalize, add_noise, plot_density, make_trajectories, set_logger, construct_default_quadtree, save, load, compute_num_params, set_budget
from dataset import TrajectoryDataset, PretrainingDataset
from models import compute_loss_generator, construct_generator
import torch.nn.functional as F
from opacus.utils.batch_memory_manager import BatchMemoryManager

from opacus import PrivacyEngine
from pytorchtools import EarlyStopping
import evaluation




def make_targets_of_all_layers(target_locations, tree):
    n_locations = len(tree.get_leafs())
    batch_size = target_locations.shape[0]
    target_locations = target_locations.view(-1)
    node_paths = [tree.state_to_node_path(state.item())[1:] for state in target_locations]
    output = []
    for node_path in node_paths:
        if None in node_path:
            target = [TrajectoryDataset.ignore_idx(n_locations) for _ in range(tree.max_depth)]
        else:
            target = [node.oned_coordinate for node in node_path]
        output.append(target)
    output_ = []
    for i in range(tree.max_depth):
        output_.append(torch.tensor([location[i] for location in output]).view(batch_size, -1).to(target_locations.device))
    return output_

def train_with_discrete_time(generator, optimizer, loss_model, input_locations, target_locations, input_times, target_times, labels, coef_location, coef_time, train_all_layers=False):
    is_dp = hasattr(generator, "module")
    # if loss_model == compute_loss_gru_meta_gru_net:
    #     target_locations = torch.tensor([generator.meta_net.tree.state_to_path(state.item()) for state in target_locations.view(-1)]).view(target_locations.shape[0], target_locations.shape[1], generator.meta_net.tree.max_depth).to(target_locations.device)
    #     output_locations, output_times = generator([input_locations, input_times], labels, target=target_locations)
    # else:
    # output_locations, output_times = generator([input_locations, input_times], labels)
    (output_locations, output_times), _ = generator([input_locations, input_times])
    if train_all_layers:
        # target_locations = make_targets_of_all_layers(target_locations, generator.meta_net.tree)
        target_locations = make_targets_of_all_layers(target_locations, generator.location_encoding_component.tree)

    # if generator.meta_net.is_consistent:
    if False:
        # same thing as the below one
        losses = []
        for i in range(len(target_locations)):
            loss_depth_i = 0
            counter_depth_i = 0
            for j in range(len(target_locations[i])):
                for k in range(len(target_locations[i][j])):
                    if target_locations[i][j][k].item() != TrajectoryDataset.ignore_idx(generator.meta_net.n_locations):
                        # new_target_locations.append(0)
                        # new_output_locations.append(output_locations[i][j][k][target_locations[i][j][k]])
                        # print(output_locations[i][j][k][target_locations[i][j][k]].view(-1,1))
                        loss_depth_i += (torch.nn.functional.nll_loss(output_locations[i][j][k][target_locations[i][j][k]].view(-1,1), torch.tensor([0])))
                        counter_depth_i += 1
            losses.append(loss_depth_i / counter_depth_i)

        losses.append(F.nll_loss(output_times.view(-1, output_times.shape[-1]), (target_times).view(-1)))

    # print(output_locations.shape, target_locations.shape, output_times.shape, target_times.shape)
    losses = loss_model(target_locations, target_times, output_locations, output_times, coef_location, coef_time)
    loss = sum(losses)
    optimizer.zero_grad()
    loss.backward()

    if is_dp:
        norms = []
        # get the norm of gradient example
        for name, param in generator.named_parameters():
            if 'grad_sample' not in vars(param):
                # in this case, the gradient is already accumulated
                # norms.append(param.grad.reshape(len(param.grad), -1).norm(2, dim=-1))
                pass
            else:
                norms.append(param.grad_sample.reshape(len(param.grad_sample), -1).norm(2, dim=-1))
        
        if len(norms[0]) > 1:
            norms = torch.stack(norms, dim=1).norm(2, dim=-1).detach().cpu().numpy()
        else:
            norms = torch.concat(norms, dim=0).detach().cpu().numpy()
    else:
        # compute the norm of gradient
        norms = []
        for name, param in generator.named_parameters():
            if param.grad is None:
                continue
            norms.append(param.grad.reshape(-1))
        norms = torch.cat(norms, dim=0)
        # print("are", norms.max(), norms.min())
        norm = norms.norm(2, dim=-1).detach().cpu().numpy()
        norms = [norm]

    optimizer.step()
    losses = [loss.item() for loss in losses]
    losses.append(np.mean(norms))

    return losses


def train_epoch(data_loader, generator, optimizer, loss_model, train_all_layers, coef_location, coef_time):
    losses = []
    device = next(generator.parameters()).device
    for i, batch in enumerate(data_loader):
        input_locations = batch["input"].to(device, non_blocking=True)
        target_locations = batch["target"].to(device, non_blocking=True)
        references = [tuple(v) for v in batch["reference"]]
        input_times = batch["time"].to(device, non_blocking=True)
        target_times = batch["time_target"].to(device, non_blocking=True)

        loss = train_with_discrete_time(generator, optimizer, loss_model, input_locations, target_locations, input_times, target_times, references, coef_location, coef_time, train_all_layers=train_all_layers)
        # print(norm)
        losses.append(loss)

    return np.mean(losses, axis=0)

def clustering(clustering_type, n_locations, logger):
    logger.info(f"clustering type: {clustering_type}")
    n_bins = int(np.sqrt(n_locations)) -2
    # if clustering_type == "distance":
        # distance_matrix = np.load(training_data_dir.parent.parent / f"distance_matrix_bin{n_bins}.npy")
        # location_to_class = evaluation.clustering(dataset.global_counts[0], distance_matrix, args.n_classes)
        # privtree = None
    # elif clustering_type == "privtree":
        # location_to_class, privtree = evaluation.privtree_clustering(dataset.global_counts[0], theta=args.privtree_theta)
    if clustering_type == "depth":
        location_to_class, privtree = depth_clustering(n_bins)
    else:
        raise NotImplementedError
    return location_to_class, privtree

# def construct_pretraining_network(clustering_type, network_type, n_locations, memory_dim, memory_hidden_dim, location_embedding_dim, multilayer, consistent, logger):

#     location_to_class, privtree = clustering(clustering_type, n_locations, logger)
#     # class needs to correspond to node 
#     n_classes = len(set(location_to_class.values()))

#     pretraining_network_class, _ = guide_to_model(network_type)
#     if network_type == "markov1":
#         pass
#         # normalize count by dim = 1
#         # target_counts = target_counts / target_counts.sum(dim=1).reshape(-1,1)
#         # generator = Markov1Generator(target_counts.cpu(), location_to_class)
#         # eval_generator = generator
#         # optimizer = None
#         # data_loader = None
#         # privacy_engine = None
#         # args.n_epochs = 0
#     elif network_type == "baseline":
#         pretraining_network = pretraining_network_class(memory_hidden_dim, memory_dim, n_locations, n_classes, "relu")
#     elif network_type == "hrnet":
#         pretraining_network = pretraining_network_class(n_locations, memory_dim, memory_hidden_dim, location_embedding_dim, privtree, "relu", multilayer=multilayer, is_consistent=consistent)

#     compute_num_params(pretraining_network, logger)
        
#     return pretraining_network, location_to_class

def prepare_transition_matrix(location_to_class, transition_type, dataset, clipping, epsilon, save_dir, logger):
    n_classes = len(set(location_to_class.values()))
    transition_matrix = []
    for i in range(n_classes):
        if transition_type == "marginal":
            logger.info(f"use marginal transition matrix")
            next_location_counts = dataset.next_location_counts
        elif transition_type == "first":
            logger.info(f"use first transition matrix")
            next_location_counts = evaluation.make_next_location_count(dataset, 0)
        elif transition_type == "test":
            logger.info(f"use test transition matrix")
            next_location_counts = {location: [1] * dataset.n_locations for location in range(dataset.n_locations)}

        # find the locations belonging to the class i
        next_location_count_i = torch.zeros(dataset.n_locations)
        locations = [location for location, class_ in location_to_class.items() if class_ == i]
        logger.info(f"n locations in class {i}: {len(locations)}")
        for location in locations:
            if location in next_location_counts:
                next_location_count_i += np.array(next_location_counts[location]) 
        logger.info(f"sum of next location counts in class {i}: {sum(next_location_count_i)} add noise by epsilon = {epsilon}")
        target_count_i = add_noise(next_location_count_i, clipping, epsilon)
        target_count_i = torch.tensor(target_count_i)
        
        transition_matrix.append(target_count_i)

        plot_density(target_count_i, dataset.n_locations, save_dir / "imgs" / f"class_next_location_distribution_{i}.png")
    
    return torch.stack(transition_matrix)

def compute_loss_for_pretraining(pretraining_network_output, target):

    multitask = type(pretraining_network_output) == list
    # multitask training
    if multitask:
        assert (type(pretraining_network_output) == list) and (type(target) == list), f"{type(pretraining_network_output)} and {type(target)} must be list because of multitask training"

        losses = []
        # for each depth
        for i in range(len(pretraining_network_output)):
            losses.append(F.kl_div(pretraining_network_output[i], target[i].to(pretraining_network_output[i].device), reduction='batchmean'))
        loss = sum(losses)
    else:
        pretraining_network_output = pretraining_network_output.view(*target.shape)
        loss = F.kl_div(pretraining_network_output, target.to(pretraining_network_output.device), reduction='batchmean')
    
    return loss



def pre_training_pretraining_network(transition_matrix, privtree, n_iter, pretraining_network, patience, save_dir, pretraining_method, logger):
    device = next(pretraining_network.parameters()).device
    
    # class_encoder converts class to a vector which is used for the input of the temp_network, and this itself is not trained
    class_encoder = pretraining_network.location_encoding_component.make_class_encoder(privtree).to(device)
    # temp_network works as the substitution of the prefix encoding component (i.e., this outputs input for the scoring component) for pre-training because a record of pre-training is not a sequence
    temp_network = pretraining_network.prefix_encoding_component.make_temp_network(class_encoder.dim).to(device)
    optimizer = optim.Adam([{"params":pretraining_network.parameters()}, {"params":temp_network.parameters()}], lr=0.001)

    # make data loader for pre-training with the transition matrix
    # pretraining_method designates the way of sampling of training data
    batch_size = 100
    pretraining_dataset = PretrainingDataset(transition_matrix, pretraining_method, n_iter, batch_size, pretraining_network)
    pretraining_data_loader = torch.utils.data.DataLoader(pretraining_dataset, num_workers=0, pin_memory=True, batch_size=batch_size, collate_fn=pretraining_dataset.make_collate_fn())

    # pre-training with early stopping
    early_stopping = EarlyStopping(patience=patience, path=save_dir / "pretraining_network.pt", delta=1e-6)
    with tqdm.tqdm(pretraining_data_loader) as pbar:
        for epoch, batch in enumerate(pbar):
            # input
            input = batch["input"].to(device)
            pretraining_network_output = pretraining_network.transition(input, class_encoder, temp_network)

            # compute loss
            loss = compute_loss_for_pretraining(pretraining_network_output, batch["target"])

            # compute gradient and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"loss: {loss.item()}")

            # early stopping
            early_stopping(loss.item(), pretraining_network)
            if early_stopping.early_stop:
                pretraining_network.load_state_dict(torch.load(save_dir / "pretraining_network.pt"))
                logger.info(f"load meta network from {save_dir / 'pretraining_network.pt'}")
                break
    logger.info(f"best loss of meta training at {epoch}: {early_stopping.best_score}")

    # test
    logger.info("save test results to " + str(save_dir / "imgs" / f"pretraining_network_output_i.png"))
    test_pretrained_network(pretraining_network, class_encoder, temp_network, len(transition_matrix), len(transition_matrix[0]), save_dir)

def test_pretrained_network(pretraining_network, class_encoder, temp_network, n_classes, n_locations, save_dir):
    device = next(pretraining_network.parameters()).device

    # plot the test output of meta_network
    with torch.no_grad():
        pretraining_network.pre_training = False
        pretraining_network.eval()
        test_input = torch.eye(n_classes).to(device)
        meta_network_output = pretraining_network.transition(test_input, class_encoder, temp_network)
        if type(meta_network_output) == list:
            meta_network_output = meta_network_output[-1]
        for i in range(n_classes):
            plot_density(torch.exp(meta_network_output[i]).cpu().view(-1), n_locations, save_dir / "imgs" / f"pretraining_network_output_{i}.png")
        pretraining_network.train()


def construct_dataset(training_data_dir, route_data_path, n_time_split):
    # load dataset config    
    with open(training_data_dir / "params.json", "r") as f:
        param = json.load(f)
    n_locations = param["n_locations"]
    dataset_name = param["dataset"]

    # load data
    trajectories = load(training_data_dir / "training_data.csv")
    time_trajectories = load(training_data_dir / "training_data_time.csv")

    # route data is optional
    # this is used for MTNet 
    route_trajectories = load(route_data_path) if route_data_path is not None else None

    return TrajectoryDataset(trajectories, time_trajectories, n_locations, n_time_split, route_data=route_trajectories, dataset_name=dataset_name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.deterministic = True

def check_hyperparameters(kwargs, dataset, logger):
    # set batch size
    if kwargs["batch_size"] == 0:
        kwargs["batch_size"] = int(np.sqrt(len(dataset)))
        logger.info("batch size is set as " + str(kwargs["batch_size"]))
    if kwargs["physical_batch_size"] == 0:
        kwargs["physical_batch_size"] = kwargs["batch_size"]
        logger.info("physical batch size is set as " + str(kwargs["physical_batch_size"]))
    if kwargs["consistent"] and not kwargs["multitask"]:
        raise ValueError("consistent is True but multitask is False")
    if kwargs["model_name"] != "hrnet" and kwargs["multitask"]:
        raise ValueError("multitask is True but model_name is not hrnet")
    if kwargs["pre_n_iter"] == 0:
        kwargs["epsilon"] = 0
        logger.info("pre-training is not done")
    else:
        if kwargs["epsilon"] == 0:
            # decide the budget for the pre-training (this is for depth_clustering with depth = 2)
            kwargs["epsilon"] = set_budget(len(dataset), int(np.sqrt(dataset.n_locations)) -2)
            logger.info(f"epsilon is set as: {kwargs['epsilon']} by our method")
        else:
            logger.info(f"epsilon is fixed as: {kwargs['epsilon']}")

def run(**kwargs):

    # set seed
    set_seed(kwargs["model_seed"])

    # set save path
    save_name = make_save_name(**kwargs)
    training_data_dir = pathlib.Path(make_training_data_path(kwargs["dataset_name"], kwargs["size"], save_name))
    save_dir = training_data_dir / make_model_name(**kwargs)
    (save_dir / "imgs").mkdir(exist_ok=True, parents=True)

    # set logger
    logger = set_logger(__name__, save_dir / "log.log")
    logger.info(f'used parameters {kwargs}')
    logger.info(f'training_data_dir: {training_data_dir}')

    # set device
    device = torch.device(f"cuda:{kwargs['cuda_number']}" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # load dataset
    logger.info(f"load training data from {training_data_dir / 'training_data.csv'}")
    logger.info(f"load time data from {training_data_dir / 'training_data_time.csv'}")
    dataset = construct_dataset(training_data_dir, None, kwargs["n_split"])
    logger.info(f"len of the dataset: {len(dataset)}")

    # check hyperparaneters
    check_hyperparameters(kwargs, dataset, logger)

    # make data loader
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=kwargs["batch_size"], collate_fn=dataset.make_padded_collate(kwargs["remove_first_value"], kwargs["remove_duplicate"]))

    # construct generator
    generator = construct_generator(kwargs["model_name"], dataset.n_locations, dataset.n_time_split+1, kwargs["location_embedding_dim"], kwargs["time_embedding_dim"], kwargs["memory_hidden_dim"], kwargs["multitask"], kwargs["consistent"])
    generator.to(device)

    # pre-training
    if kwargs["pre_n_iter"] != 0:
        # classify locations according to the clustering type with location semantics without dataset
        location_to_class, privtree = clustering(kwargs['clustering'], dataset.n_locations, logger)
        # prepare (DP) transition matrix
        transition_matrix = prepare_transition_matrix(location_to_class, kwargs["transition_type"], dataset, kwargs["clipping_for_transition_matrix"], kwargs["epsilon"], save_dir, logger)
        # pre-training with the transition matrix
        pre_training_pretraining_network(transition_matrix, privtree, kwargs["pre_n_iter"], generator, kwargs["pre_training_patience"], save_dir, kwargs["pretraining_method"], logger)

    # set optimizer
    optimizer = optim.Adam(generator.parameters(), lr=kwargs["learning_rate"])
    # make generator, optimizer, and data_loader private if is_dp
    if kwargs["is_dp"]:
        logger.info("privating the model")
        privacy_engine = PrivacyEngine(accountant=kwargs["accountant_mode"])
        generator, optimizer, data_loader = privacy_engine.make_private(module=generator, optimizer=optimizer, data_loader=data_loader, noise_multiplier=kwargs["noise_multiplier"], max_grad_norm=kwargs["clipping_bound"])
        eval_generator = generator._module
    else:
        logger.info("not privating the model")
        eval_generator = generator

    # traning the generator with early stopping
    early_stopping = EarlyStopping(patience=kwargs["patience"], verbose=True, path=save_dir / "checkpoint.pt", trace_func=logger.info)
    logger.info(f"early stopping patience: {kwargs['patience']}")
    for epoch in tqdm.tqdm(range(kwargs["n_epochs"])):

        # save model
        logger.info(f"save model to {save_dir / f'model_{epoch}.pt'}")
        torch.save(eval_generator.state_dict(), save_dir / f"model_{epoch}.pt")

        # training
        if not kwargs["is_dp"]:
            losses = train_epoch(data_loader, generator, optimizer, compute_loss_generator, kwargs["multitask"], kwargs["coef_location"], kwargs["coef_time"])
            epsilon = 0
        else:
            with BatchMemoryManager(data_loader=data_loader, max_physical_batch_size=min([kwargs["physical_batch_size"], kwargs["batch_size"]]), optimizer=optimizer) as new_data_loader:
                losses = train_epoch(new_data_loader, generator, optimizer, compute_loss_generator, kwargs["multitask"], kwargs["coef_location"], kwargs["coef_time"])
            epsilon = privacy_engine.get_epsilon(kwargs["dp_delta"])

        # early stopping
        early_stopping(np.sum(losses[:-1]), eval_generator)
        logger.info(f'epoch: {early_stopping.epoch} epsilon: {epsilon} | best loss: {early_stopping.best_score} | current loss: location {losses[:-2]}, time {losses[-2]}, norm {losses[-1]}')
        if early_stopping.early_stop:
            break
    
    # save parameters
    logger.info(f"save param to {save_dir / 'params.json'}")
    with open(save_dir / "params.json", "w") as f:
        json.dump(kwargs, f)