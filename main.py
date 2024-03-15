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
from models import compute_loss_meta_gru_net, compute_loss_gru_meta_gru_net, Markov1Generator, MetaGRUNet, MetaNetwork, FullLinearQuadTreeNetwork, guide_to_model
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
    if loss_model == compute_loss_gru_meta_gru_net:
        target_locations = torch.tensor([generator.meta_net.tree.state_to_path(state.item()) for state in target_locations.view(-1)]).view(target_locations.shape[0], target_locations.shape[1], generator.meta_net.tree.max_depth).to(target_locations.device)
        output_locations, output_times = generator([input_locations, input_times], labels, target=target_locations)
    else:
        output_locations, output_times = generator([input_locations, input_times], labels)
        if train_all_layers:
            target_locations = make_targets_of_all_layers(target_locations, generator.meta_net.tree)

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

def clustering(clustering_type, n_locations):
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

def construct_pretraining_network(clustering_type, network_type, n_locations, memory_dim, memory_hidden_dim, location_embedding_dim, multilayer, consistent, logger):

    logger.info(f"clustering type: {clustering_type}")
    location_to_class, privtree = clustering(clustering_type, n_locations)
    # class needs to correspond to node 
    n_classes = len(set(location_to_class.values()))

    pretraining_network_class, _ = guide_to_model(network_type)
    if network_type == "markov1":
        pass
        # normalize count by dim = 1
        # target_counts = target_counts / target_counts.sum(dim=1).reshape(-1,1)
        # generator = Markov1Generator(target_counts.cpu(), location_to_class)
        # eval_generator = generator
        # optimizer = None
        # data_loader = None
        # privacy_engine = None
        # args.n_epochs = 0
    elif network_type == "baseline":
        pretraining_network = pretraining_network_class(memory_hidden_dim, memory_dim, n_locations, n_classes, "relu")
    elif network_type == "hrnet":
        pretraining_network = pretraining_network_class(n_locations, memory_dim, memory_hidden_dim, location_embedding_dim, privtree, "relu", multilayer=multilayer, is_consistent=consistent)

    compute_num_params(pretraining_network, logger)
        
    return pretraining_network, location_to_class

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
    
    return transition_matrix

def pre_training_pretraining_network(transition_matrix, n_iter, pretraining_network, patience, save_dir, pretraining_method, logger):
    # print(len(transition_matrix), transition_matrix[0].shape)
    device = next(pretraining_network.parameters()).device
    transition_matrix = torch.stack(transition_matrix)
    # if args.meta_network_load_path == "None":
    early_stopping = EarlyStopping(patience=patience, path=save_dir / "pretraining_network.pt", delta=1e-6)
    # train_pretraining_network(pretraining_network, transition_matrix, pre_n_iter, early_stopping, pretraining_method)
    #     args.meta_network_load_path = str(save_dir / "meta_network.pt")
    # else:
    #     pretraining_network.load_state_dict(torch.load(args.meta_network_load_path))
    #     logger.info(f"load meta network from {args.meta_network_load_path}")
    # def train_pretraining_network(pretraining_network, next_location_counts, n_iter, early_stopping, distribution, logger):
    # device = next(iter(pretraining_network.parameters())).device

    optimizer = optim.Adam(pretraining_network.parameters(), lr=0.001)
    batch_size = 100
    # n_classes = pretraining_network.n_classes
    # n_locations = len(transition_matrix[0])
    # epoch = 0

    # n_locations = len(transition_matrix[0])
    # n_bins = int(np.sqrt(n_locations)) -2
    # tree = construct_default_quadtree(n_bins)
    # tree.make_self_complete()

    pretraining_dataset = PretrainingDataset(transition_matrix, pretraining_method, n_iter, batch_size, pretraining_network.name)
    pretraining_data_loader = torch.utils.data.DataLoader(pretraining_dataset, num_workers=0, pin_memory=True, batch_size=batch_size, collate_fn=pretraining_dataset.make_collate_fn())

    with tqdm.tqdm(pretraining_data_loader) as pbar:
        for epoch, batch in enumerate(pbar):
            input = batch["input"].to(device)
            target = batch["target"].to(device)

            pretraining_network_output = pretraining_network(input).view(*target.shape)

            loss = F.kl_div(pretraining_network_output, target, reduction='batchmean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # pbar.set_description(f"loss: {loss.item()} ({[v.item() for v in loss]})")
            pbar.set_description(f"loss: {loss.item()}")
            early_stopping(loss.item(), pretraining_network)

            if early_stopping.early_stop:
                pretraining_network.load_state_dict(torch.load(save_dir / "pretraining_network.pt"))
                logger.info(f"load meta network from {save_dir / 'pretraining_network.pt'}")
                break

    # def depth_to_ids(depth):
    #     return [node.id for node in tree.get_nodes(depth)]
    
    # original_targets = torch.zeros(n_classes, n_locations).to(device)
    # for i in range(n_classes):
    #     original_targets[i] = torch.tensor(transition_matrix[i])
    #     original_targets[i][original_targets[i] < 0] = 0
    #     original_targets[i] = original_targets[i] / original_targets[i].sum()

    # if pretraining_method == "eye":
    #     input = torch.eye(n_classes).to(device)

    #     # normalize
    #     input = input / input.sum(dim=1).reshape(-1,1)
    #     # target is the distribution generated by sum of next_location_distributions weighted by input
    #     target = torch.zeros(input.shape[0], n_locations).to(device)
    #     for i in range(n_classes):
    #         target += input[:,i].reshape(-1,1) * transition_matrix[i]
    #     # normalize target
    #     target[target < 0] = 0
    #     target = target / target.sum(dim=1).reshape(-1,1)
    #     target = tree.make_quad_distribution(target)

    # with tqdm.tqdm(range(n_iter)) as pbar:
    #     for epoch in pbar:
    #         # make input: (batch_size, n_classes)
    #         # input is sampled from Dirichlet distribution
    #         if pretraining_method == "dirichlet":
    #             input = torch.distributions.dirichlet.Dirichlet(torch.ones(n_classes)).sample((batch_size,)).to(device)
    #             print(input.shape)

    #             # normalize
    #             # input = input / input.sum(dim=1).reshape(-1,1)
    #             # target is the distribution generated by sum of next_location_distributions weighted by input
    #             target = torch.zeros(input.shape[0], n_locations).to(device)
    #             for i in range(n_classes):
    #                 target += input[:,i].reshape(-1,1) * transition_matrix[i]
    #             # normalize target
    #             target[target < 0] = 0
    #             target = target / target.sum(dim=1).reshape(-1,1)
    #         elif pretraining_method == "eye":
    #             input = torch.eye(n_classes).to(device)
    #         elif pretraining_method == "both":
    #             input = torch.distributions.dirichlet.Dirichlet(torch.ones(n_classes)).sample((batch_size,)).to(device)
    #             input = torch.cat([input, torch.eye(n_classes).to(device)], dim=0)
    #             target = torch.zeros(input.shape[0], n_locations).to(device)
    #             for i in range(n_classes):
    #                 target += input[:,i].reshape(-1,1) * transition_matrix[i]
    #             target[target < 0] = 0
    #             target = target / target.sum(dim=1).reshape(-1,1)
    #         else:
    #             raise NotImplementedError
            
    #         losses = []
    #         loss = 0

    #         pretraining_network_output = pretraining_network(input)
    #         print(pretraining_network_output.shape)
    #         if type(pretraining_network_output) == list:
    #             batch_size = pretraining_network_output[0].shape[0]
    #             test_target = evaluation.make_target_distributions_of_all_layers(target, tree)
    #             train_all_layers = True
    #             if train_all_layers:
    #                 # pretraining_network_output = pretraining_network.to_location_distribution(pretraining_network_output, target_depth=0)
    #                 for depth in range(tree.max_depth):
    #                     losses.append(F.kl_div(pretraining_network_output[depth].view(batch_size,-1), test_target[depth], reduction='batchmean'))
    #             else:
    #                 pretraining_network_output = pretraining_network.to_location_distribution(pretraining_network_output, target_depth=-1)
    #                 losses.append(F.kl_div(pretraining_network_output.view(batch_size,-1), test_target[-1], reduction='batchmean'))
    #             loss = sum(losses)
    #         else:
    #             quad_loss = (pretraining_network.name == "hrnet")
    #             if quad_loss:
    #                 if pretraining_method != "eye":
    #                     target = tree.make_quad_distribution(target)
    #                 pretraining_network_output = pretraining_network_output.view(*target.shape)
    #                 for depth in range(tree.max_depth):
    #                     ids = depth_to_ids(depth)
    #                     losses.append(F.kl_div(pretraining_network_output[:,ids,:], target[:,ids,:], reduction='batchmean') * 4**(tree.max_depth-depth-1))
    #             else:
    #                 pretraining_network_output = pretraining_network_output.view(*target.shape)
    #                 print(target.shape)
    #                 losses.append(F.kl_div(pretraining_network_output, target, reduction='batchmean'))
    #         # loss = compute_loss_meta_quad_tree_attention_net(pretraining_network_output, target, pretraining_network.tree)
    #         loss = sum(losses)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()


            # pbar.set_description(f"loss: {loss.item()} ({[v.item() for v in losses]})")
            # early_stopping(loss.item(), pretraining_network)

            # if early_stopping.early_stop:
            #     pretraining_network.load_state_dict(torch.load(save_dir / "pretraining_network.pt"))
            #     logger.info(f"load meta network from {save_dir / 'pretraining_network.pt'}")
            #     break

    logger.info(f"best loss of meta training at {epoch}: {early_stopping.best_score}")

 
    # test
    logger.info("save test output to " + str(save_dir / "imgs" / f"pretraining_network_output_i.png"))
    test_pretrained_network(pretraining_network, len(transition_matrix[0]), save_dir)

    return pretraining_network

def test_pretrained_network(pretraining_network, n_locations, save_dir):
    n_classes = pretraining_network.n_classes
    device = next(pretraining_network.parameters()).device

    # plot the test output of meta_network
    with torch.no_grad():
        pretraining_network.pre_training = False
        pretraining_network.eval()
        test_input = torch.eye(n_classes).to(device)
        meta_network_output = pretraining_network(test_input)
        if type(meta_network_output) == list:
            meta_network_output = meta_network_output[-1]
        for i in range(n_classes):
            plot_density(torch.exp(meta_network_output[i]).cpu().view(-1), n_locations, save_dir / "imgs" / f"pretraining_network_output_{i}.png")
        pretraining_network.train()

def construct_generator(n_locations, meta_network, network_type, location_embedding_dim, n_split, trajectory_type_dim, hidden_dim, reference_to_label, logger):

    _, generator_class = guide_to_model(network_type)

    # time_dim is n_time_split + 2 (because of the edges 0 and >max)
    generator = generator_class(meta_network, n_locations, location_embedding_dim, n_split+2, trajectory_type_dim, hidden_dim, reference_to_label)
    compute_num_params(generator, logger)
    
    return generator, compute_loss_meta_gru_net

def construct_dataset(training_data_dir, route_data_path, n_time_split):

    # load dataset config    
    with open(training_data_dir / "params.json", "r") as f:
        param = json.load(f)
    n_locations = param["n_locations"]
    dataset_name = param["dataset"]

    trajectories = load(training_data_dir / "training_data.csv")
    if route_data_path is not None:
        # try:
        route_trajectories = load(route_data_path)
        # except:
            # print("failed to load route data", route_data_path)
            # route_trajectories = None        
    else:
        route_trajectories = None
    time_trajectories = load(training_data_dir / "training_data_time.csv")

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
        
    kwargs["consistent"] = kwargs["consistent"] and kwargs["train_all_layers"]
    if kwargs["consistent"] and not kwargs["train_all_layers"]:
        kwargs["consistent"] = False
        logger.info("!!!!!! consistent is set as False because train_all_layers is False")
    if kwargs["model_name"] != "hrnet":
        kwargs["train_all_layers"] = False

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

    # prepare and private pre-train the temp model
    pretraining_network, location_to_class = construct_pretraining_network(kwargs["clustering"], kwargs["model_name"], dataset.n_locations, kwargs["memory_dim"], kwargs["memory_hidden_dim"], kwargs["location_embedding_dim"], kwargs["multilayer"], kwargs["consistent"], logger)
    pretraining_network.to(device)
    if kwargs["pre_n_iter"] != 0:
        # prepare (DP) transition matrix
        transition_matrix = prepare_transition_matrix(location_to_class, kwargs["transition_type"], dataset, kwargs["clipping_for_transition_matrix"], kwargs["epsilon"], save_dir, logger)
        # pre-training with the transition matrix
        pretraining_network = pre_training_pretraining_network(transition_matrix, kwargs["pre_n_iter"], pretraining_network, kwargs["pre_training_patience"], save_dir, kwargs["pretraining_method"], logger)
    # remove the temp component for pre-training
    if hasattr(pretraining_network, "remove_class_to_query"):
        pretraining_network.remove_class_to_query()


    # prepare the generator
    generator, loss_model = construct_generator(dataset.n_locations, pretraining_network, kwargs["model_name"], kwargs["location_embedding_dim"], kwargs["n_split"], len(dataset.label_to_reference), kwargs["hidden_dim"], dataset.reference_to_label, logger)
    kwargs["num_params"] = compute_num_params(generator, logger)
    generator.to(device)
    optimizer = optim.Adam(generator.parameters(), lr=kwargs["learning_rate"])

    # make private if is_dp
    if kwargs["is_dp"]:
        logger.info("privating the model")
        privacy_engine = PrivacyEngine(accountant=kwargs["accountant_mode"])
        generator, optimizer, data_loader = privacy_engine.make_private(module=generator, optimizer=optimizer, data_loader=data_loader, noise_multiplier=kwargs["noise_multiplier"], max_grad_norm=kwargs["clipping_bound"])
        eval_generator = generator._module
    else:
        logger.info("not privating the model")
        eval_generator = generator

    early_stopping = EarlyStopping(patience=kwargs["patience"], verbose=True, path=save_dir / "checkpoint.pt", trace_func=logger.info)
    logger.info(f"early stopping patience: {kwargs['patience']}, save path: {save_dir / 'checkpoint.pt'}")

    # traning the generator
    for epoch in tqdm.tqdm(range(kwargs["n_epochs"])):

        # save model
        logger.info(f"save model to {save_dir / f'model_{epoch}.pt'}")
        torch.save(eval_generator.state_dict(), save_dir / f"model_{epoch}.pt")

        # training
        if not kwargs["is_dp"]:
            losses = train_epoch(data_loader, generator, optimizer, loss_model, kwargs["train_all_layers"], kwargs["coef_location"], kwargs["coef_time"])
            epsilon = 0
        else:
            with BatchMemoryManager(data_loader=data_loader, max_physical_batch_size=min([kwargs["physical_batch_size"], kwargs["batch_size"]]), optimizer=optimizer) as new_data_loader:
                losses = train_epoch(new_data_loader, generator, optimizer, loss_model, kwargs["train_all_layers"], kwargs["coef_location"], kwargs["coef_time"])
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