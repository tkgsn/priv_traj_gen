import argparse
import random
import numpy as np
import torch
import tqdm
from torch import nn, optim
import json
from scipy.spatial.distance import jensenshannon

from my_utils import get_datadir, clustering, privtree_clustering, noise_normalize, add_noise, plot_density, make_trajectories, set_logger, construct_default_quadtree, save, load, compute_num_params
from dataset import TrajectoryDataset
from models import compute_loss_meta_gru_net, compute_loss_gru_meta_gru_net, Markov1Generator, MetaGRUNet, MetaNetwork, FullLinearQuadTreeNetwork
import torch.nn.functional as F
from opacus.utils.batch_memory_manager import BatchMemoryManager

from opacus import PrivacyEngine
from pytorchtools import EarlyStopping

from evaluation import count_source_locations, count_target_locations, compute_distribution_from_count, compute_route_count, compute_distance_distribution, compute_destination_count, compute_next_location_distribution, compute_global_distribution, make_target_distributions_of_all_layers, evaluate_next_location_on_test_dataset, compute_distribution_js_for_each_depth
from data_post_processing import post_process_chengdu

def evaluate(epoch):

    if epoch % args.eval_interval != 0:
        return
    if epoch == 0 and not args.eval_initial:
        return
    # clipped_trajectories = global_clipping(trajectories, args.global_clip)
    # clipped_trajectories = trajectories
    next_location_counts = dataset.next_location_counts
    first_next_location_counts = dataset.first_next_location_counts
    second_next_location_counts = dataset.second_next_location_counts
    global_counts = dataset.global_counts

    next_location_distributions = {key: noise_normalize(next_location_count) for key, next_location_count in next_location_counts.items()}
    first_next_location_distributions = {key: noise_normalize(first_next_location_count) for key, first_next_location_count in first_next_location_counts.items()}
    second_next_location_distributions = {key: noise_normalize(second_next_location_count) for key, second_next_location_count in second_next_location_counts.items()}
    global_distributions = [noise_normalize(global_count) for global_count in global_counts]
    noisy_global_distributions = [noise_normalize(add_noise(global_count, args.global_clip+1, args.epsilon)) for global_count in global_counts]
    for i in range(len(noisy_global_distributions)):
        if noisy_global_distributions[i] == None:
            noisy_global_distributions[i] = [1/dataset.n_locations] * dataset.n_locations
    noisy_global_distributions = torch.tensor(noisy_global_distributions)
    n_test_locations = min(args.n_test_locations, sum(np.array(global_counts[0])>0))
    # logger.info(f"n_test_locations is set as min of args.n_test_locations and the number of locations appearing as the base location {args.n_test_locations}, {sum(np.array(global_counts[0])>0)}")
    top_base_locations = np.argsort(global_counts[0])[::-1]
    param["test_locations"] = top_base_locations[:n_test_locations].tolist()
    test_traj, test_traj_time = make_test_data(data_path, top_base_locations, n_test_locations, args.dataset)
    test_dataset = TrajectoryDataset(test_traj, test_traj_time, n_locations, args.n_split)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, shuffle=False, pin_memory=True, batch_size=args.batch_size, collate_fn=test_dataset.make_padded_collate(args.remove_first_value))

    param["global_distributions"] = noisy_global_distributions.tolist()
    distance_matrix = np.load(data_path.parent.parent / f"distance_matrix_bin{int(np.sqrt(n_locations)) -2}.npy")

    generator.eval()
    with torch.no_grad():
        results = {}

        if args.evaluate_first_next_location:
            jss = evaluate_next_location_on_test_dataset(first_next_location_distributions, test_data_loader, eval_generator, 1)
            results["first_next_location_js"] = jss


        if args.evaluate_second_next_location and (dataset.seq_len > 2):
            jss = evaluate_next_location_on_test_dataset(second_next_location_distributions, test_data_loader, eval_generator, 2)
            results["second_next_location_js"] = jss


        if args.evaluate_global or args.evaluate_source or args.evaluate_target or args.evaluate_route or args.evaluate_destination or args.evaluate_distance:
            generated = eval_generator.make_sample(dataset.references, min([1000, len(dataset.references)]))
            if len(generated) == 2:
                # when n_data == 2 it causes bug
                generated_trajs, generated_time_trajs = generated
                generated_time_trajs = dataset.convert_time_label_trajs_to_time_trajs(generated_time_trajs)
            else:
                generated_trajs = generated
                generated_time_trajs = dataset.time_label_trajs
            if (args.dataset == "chengdu") and args.post_process:
                generated_trajs = post_process_chengdu(generated_trajs)

        if args.evaluate_global:
            gene_global_distributions = [compute_global_distribution(generated_trajs, generated_time_trajs, time, dataset.n_locations, dataset._time_to_label) for time in [i[0] for i in dataset.time_ranges]]
            global_jss = []
            for i in range(len(dataset.time_ranges)):
                if gene_global_distributions[i] is None or global_distributions[i] is None:
                    global_jss.append(1)
                else:
                    js = jensenshannon(gene_global_distributions[i], global_distributions[i])**2
                    global_jss.append(js)
            results["global_js"] = global_jss

        if args.evaluate_source:
            count = count_source_locations(trajectories)
            original_source_distribution = compute_distribution_from_count(count, dataset.n_locations)
            count = count_source_locations(generated_trajs)
            generated_source_distribution = compute_distribution_from_count(count, dataset.n_locations)
            source_js = jensenshannon(original_source_distribution, generated_source_distribution)**2
            results["source_js"] = source_js

        if args.evaluate_target:
            count = count_target_locations(trajectories)
            original_target_distribution = compute_distribution_from_count(count, dataset.n_locations)
            count = count_target_locations(generated_trajs)
            generated_target_distribution = compute_distribution_from_count(count, dataset.n_locations)
            target_js = jensenshannon(original_target_distribution, generated_target_distribution)**2
            results["target_js"] = target_js

        if args.evaluate_route:
            route_jss = []
            for location in top_base_locations[:n_test_locations]:
                count = compute_route_count(trajectories, location)
                original_route_distribution = compute_distribution_from_count(count, dataset.n_locations)
                count = compute_route_count(generated_trajs, location)
                generated_route_distribution = compute_distribution_from_count(count, dataset.n_locations)
                route_js = jensenshannon(original_route_distribution, generated_route_distribution)**2
                route_jss.append(route_js)
            results["route_js"] = route_jss

        if args.evaluate_destination:
            destination_jss = []
            for location in top_base_locations[:n_test_locations]:
                count = compute_destination_count(trajectories, location)
                original_destination_distribution = compute_distribution_from_count(count, dataset.n_locations)
                count = compute_destination_count(generated_trajs, location)
                generated_destination_distribution = compute_distribution_from_count(count, dataset.n_locations)
                destination_js = jensenshannon(original_destination_distribution, generated_destination_distribution)**2
                destination_jss.append(destination_js)
            results["destination_js"] = destination_jss

        if args.evaluate_empirical_next_location:
            next_location_jss = []
            generated_next_location_distributions = []
            target_next_location_distributions = []
            for location in top_base_locations[:n_test_locations]:
                generated_next_location_distribution = compute_next_location_distribution(location, generated_trajs, dataset.n_locations)
                if generated_next_location_distribution is None:
                    logger.info(f"WARNING: this base location {location} does not appear in the generated dataset.")
                else:
                    generated_next_location_distributions.append(generated_next_location_distribution)
                    target_next_location_distributions.append(next_location_distributions[location])

            next_location_jss.append(compute_distribution_js_for_each_depth(torch.tensor(generated_next_location_distributions), torch.tensor(target_next_location_distributions)))
            results["empirical_next_location_js"] = next_location_jss
        
        if args.evaluate_distance:
            n_bins = 100
            original_distance_distribution = compute_distance_distribution(distance_matrix, trajectories, n_bins)
            generated_distance_distribution = compute_distance_distribution(distance_matrix, generated_trajs, n_bins)
            distance_js = jensenshannon(original_distance_distribution, generated_distance_distribution)**2
            results["distance_js"] = distance_js


    logger.info(f"evaluation: {results}")

    resultss.append(results)
    args.end_epoch = early_stopping.epoch
    args.end_loss = early_stopping.best_score
    args.results = resultss

    param.update(vars(args))
    logger.info(f"save param to {save_path / 'params.json'}")
    with open(save_path / "params.json", "w") as f:
        json.dump(param, f)
    
    generator.train()


def train_meta_network(meta_network, next_location_distributions, n_iter, early_stopping, distribution="dirichlet"):
    device = next(iter(meta_network.parameters())).device
    optimizer = optim.Adam(meta_network.parameters(), lr=0.001)
    n_classes = meta_network.n_classes
    n_locations = len(next_location_distributions[0])
    batch_size = 100
    epoch = 0
    n_bins = int(np.sqrt(n_locations)) -2
    tree = construct_default_quadtree(n_bins)
    tree.make_self_complete()

    def depth_to_ids(depth):
        return [node.id for node in tree.get_nodes(depth)]
    # make test data
    test_input = torch.eye(n_classes).to(device)
    
    with tqdm.tqdm(range(n_iter)) as pbar:
        for epoch in pbar:
            # make input: (batch_size, n_classes)
            # input is sampled from Dirichlet distribution
            if distribution == "dirichlet":
                input = torch.distributions.dirichlet.Dirichlet(torch.ones(n_classes)).sample((batch_size,)).to(device)
            elif distribution == "eye":
                input = torch.eye(n_classes).to(device)
            else:
                raise NotImplementedError

            # normalize
            input = input / input.sum(dim=1).reshape(-1,1)
            meta_network_output = meta_network(input)
            # target is the distribution generated by sum of next_location_distributions weighted by input
            target = torch.zeros(input.shape[0], n_locations).to(device)
            for i in range(n_classes):
                target += input[:,i].reshape(-1,1) * next_location_distributions[i]
            
            losses = []
            loss = 0
            if type(meta_network_output) == list:
                quad_loss = False
                if quad_loss:
                    target = tree.make_quad_distribution(target)
                    meta_network_output = meta_network_output.view(*target.shape)
                    for depth in range(tree.max_depth):
                        ids = depth_to_ids(depth)
                        loss += F.kl_div(meta_network_output[:,ids,:], target[:,ids,:], reduction='batchmean') * 4**(tree.max_depth-depth-1)
                else:
                    batch_size = meta_network_output[0].shape[0]
                    test_target = make_target_distributions_of_all_layers(target, tree)
                    train_all_layers = True
                    if train_all_layers:
                        # meta_network_output = meta_network.to_location_distribution(meta_network_output, target_depth=0)
                        for depth in range(tree.max_depth):
                            losses.append(F.kl_div(meta_network_output[depth].view(batch_size,-1), test_target[depth], reduction='batchmean'))
                    else:
                        meta_network_output = meta_network.to_location_distribution(meta_network_output, target_depth=-1)
                        losses.append(F.kl_div(meta_network_output.view(batch_size,-1), test_target[-1], reduction='batchmean'))
                    loss = sum(losses)
            else:
                meta_network_output = meta_network_output.view(*target.shape)
                losses.append(F.kl_div(meta_network_output, target, reduction='batchmean'))
            # loss = compute_loss_meta_quad_tree_attention_net(meta_network_output, target, meta_network.tree)
            loss = sum(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        

            # test
            with torch.no_grad():
                meta_network_output = meta_network(test_input)
                meta_network_output = meta_network_output[-1] if type(meta_network_output) == list else meta_network_output
                loss = F.kl_div(meta_network_output[:n_classes].view(-1, n_locations), next_location_distributions, reduction='batchmean')
                # loss = compute_loss_meta_quad_tree_attention_net(meta_network_output, next_location_distributions, meta_network.tree, True)

            pbar.set_description(f"loss: {loss.item()} ({[v.item() for v in losses]})")
            early_stopping(loss.item(), meta_network)

            if early_stopping.early_stop:
                meta_network.load_state_dict(torch.load(save_path / "meta_network.pt"))
                logger.info(f"load meta network from {save_path / 'meta_network.pt'}")
                break

    logger.info(f"best loss of meta training at {epoch}: {early_stopping.best_score}")



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


def make_test_data(data_path, top_base_locations, n_test_location, dataset_name):
    n_test_location = min(n_test_location, len(top_base_locations))
    test_data_dir = data_path
    if (test_data_dir / "test_data.csv").exists() and (test_data_dir / "test_data_time.csv").exists():
        test_traj = load_dataset(test_data_dir / "test_data.csv", logger=logger)
        test_traj_time = load_dataset(test_data_dir / "test_data_time.csv", logger=logger)
    else:
        pad = np.ones((n_test_location, 1), dtype=int) * len(top_base_locations)
        if dataset_name == "chengdu":
            test_traj = np.concatenate([np.array(top_base_locations[:n_test_location]).reshape(-1,1), pad], axis=1).tolist()
            test_traj_time = [[0, 800, 1439]]*n_test_location
        elif dataset_name == "taxi" or dataset_name == "random":
            test_traj = np.concatenate([np.array(top_base_locations[:n_test_location]).reshape(-1,1), pad], axis=1).tolist()
            test_traj_time = [[0, 1]]*n_test_location
        elif dataset_name == "rotation":
            first_locations = np.array(top_base_locations[:n_test_location])
            tree = construct_default_quadtree(int(np.sqrt(n_locations)) -2)
            tree.make_self_complete()
            first_locations_ = []
            second_locations = []
            for first_location in first_locations:
                first_location_id_in_the_depth = tree.get_location_id_in_the_depth(first_location, 2)
                second_location_id_in_the_depth = first_location_id_in_the_depth + 1
                second_location_node_id = tree.node_id_to_hidden_id.index(second_location_id_in_the_depth)
                second_location_node = tree.get_node_by_id(second_location_node_id)
                second_location_candidates = second_location_node.state_list
                # choose the second location that appears the most
                second_location = max(second_location_candidates, key=lambda x: sum(dataset.second_next_location_counts[x]))
                if sum(dataset.second_next_location_counts[second_location]) == 0:
                    print("this area does not appear as the second location")
                    continue
                first_locations_.append(first_location)
                second_locations.append(second_location)
            first_locations = np.array(first_locations_)
            second_locations = np.array(second_locations)
            test_traj = np.concatenate([first_locations.reshape(-1,1), second_locations.reshape(-1,1), pad], axis=1).tolist()
            test_traj_time = [[0, 1, 2]]*n_test_location
        else:
            test_traj = np.concatenate([np.array(top_base_locations[:n_test_location]).reshape(-1,1), pad, np.array(top_base_locations[:n_test_location]).reshape(-1,1)], axis=1).tolist()
            test_traj_time = [[0, 800, 1200, 1439]]*n_test_location

    return test_traj, test_traj_time

def train_epoch(data_loader, generator, optimizer):
    losses = []
    for i, batch in enumerate(data_loader):
        input_locations = batch["input"].cuda(args.cuda_number, non_blocking=True)
        target_locations = batch["target"].cuda(args.cuda_number, non_blocking=True)
        references = [tuple(v) for v in batch["reference"]]
        input_times = batch["time"].cuda(args.cuda_number, non_blocking=True)
        target_times = batch["time_target"].cuda(args.cuda_number, non_blocking=True)

        loss = train_with_discrete_time(generator, optimizer, loss_model, input_locations, target_locations, input_times, target_times, references, args.coef_location, args.coef_time, train_all_layers=args.train_all_layers)
        # print(norm)
        losses.append(loss)

    return np.mean(losses, axis=0)

def construct_generator(data_loader):

    distance_matrix = np.load(data_path.parent.parent / f"distance_matrix_bin{int(np.sqrt(n_locations)) -2}.npy")

    target_next_location_distributions = []
    numpy_target_next_location_distributions = []

    # clustering
    print("WARNING: clustering is done by real data")
    if args.clustering == "privtree":
        logger.info(f"use privtree clustering by {args.privtree_theta}")
        location_to_class, privtree = privtree_clustering(dataset.global_counts[0], theta=args.privtree_theta)
    else:
        logger.info("use distance clustering")
        location_to_class = clustering(dataset.global_counts[0], distance_matrix, args.n_classes)
    args.n_classes = len(set(location_to_class.values()))
    
    for i in range(args.n_classes):
        if args.transition_type == "marginal":
            logger.info(f"use marginal transition matrix")
            next_location_counts = dataset.next_location_counts
        elif args.transition_type == "first":
            logger.info(f"use first transition matrix")
            next_location_counts = dataset.first_next_location_counts
        # find the locations belonging to the class i
        next_location_distribution_i = torch.zeros(dataset.n_locations)
        locations = [location for location, class_ in location_to_class.items() if class_ == i]
        logger.info(f"n locations in class {i}: {len(locations)}")
        for location in locations:
            next_location_distribution_i += np.array(next_location_counts[location])
        logger.info(f"sum of next location counts in class {i}: {sum(next_location_distribution_i)}")
        real_target_distribution = noise_normalize(next_location_distribution_i)
        if real_target_distribution is None:
            real_target_distribution = torch.zeros(dataset.n_locations)
            real_target_distribution += 1/dataset.n_locations
            target_distribution = real_target_distribution
        else:
            target_distribution = noise_normalize(add_noise(next_location_distribution_i, args.global_clip+1, args.epsilon))
            # compute js
            js = jensenshannon(real_target_distribution, target_distribution)**2
            logger.info(f"js divergence for class {i}: {js}")

        next_location_distribution_i = torch.tensor(target_distribution)
        
        target_next_location_distributions.append(next_location_distribution_i)
        numpy_target_next_location_distributions.append(next_location_distribution_i)

        plot_density(next_location_distribution_i, dataset.n_locations, save_path / "imgs" / f"class_next_location_distribution_{i}.png")

    # for i in range(args.n_split):
    #     target_next_location_distributions.append(noisy_global_distributions[i])
    target_next_location_distributions = torch.stack(target_next_location_distributions).cuda(args.cuda_number)

    if args.network_type == "markov1":
        generator = Markov1Generator(target_next_location_distributions.cpu(), location_to_class)
        eval_generator = generator
        optimizer = None
        data_loader = None
        privacy_engine = None
        args.n_epochs = 0
    else:
        if args.meta_n_iter == 0:
            args.epsilon = 0

        if args.network_type == "meta_network":
            meta_network = MetaNetwork(args.memory_hidden_dim, args.memory_dim, dataset.n_locations, args.n_classes, "relu").cuda(args.cuda_number)
            args.train_all_layers = False
        elif args.network_type == "fulllinear_quadtree":
            meta_network = FullLinearQuadTreeNetwork(dataset.n_locations, args.memory_dim, args.memory_hidden_dim, args.location_embedding_dim, privtree, "relu").cuda(args.cuda_number)
    
        param["n_params_meta_network"] = compute_num_params(meta_network, logger)
        
        if args.meta_network_load_path == "None":
            early_stopping = EarlyStopping(patience=args.meta_patience, path=save_path / "meta_network.pt", delta=1e-6)
            train_meta_network(meta_network, target_next_location_distributions, args.meta_n_iter, early_stopping, args.meta_dist)
            args.meta_network_load_path = str(save_path / "meta_network.pt")
        else:
            meta_network.load_state_dict(torch.load(args.meta_network_load_path))
            logger.info(f"load meta network from {args.meta_network_load_path}")
        # if meta__network has remove_embeddings_query method, remove the embeddings
        if hasattr(meta_network, "remove_class_to_query"):
            meta_network.remove_class_to_query()

        generator = MetaGRUNet(meta_network, n_locations, args.location_embedding_dim, args.n_split+3, len(dataset.label_to_reference), args.hidden_dim, dataset.reference_to_label).cuda(args.cuda_number)
        param["n_params_generator"] = compute_num_params(generator, logger)

        optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate)
        if args.is_dp:
            privacy_engine = PrivacyEngine(accountant=args.accountant_mode)
            generator, optimizer, data_loader = privacy_engine.make_private(module=generator, optimizer=optimizer, data_loader=data_loader, noise_multiplier=args.noise_multiplier, max_grad_norm=args.clipping_bound)
            eval_generator = generator._module
        else:
            eval_generator = generator
    
    return generator, eval_generator, compute_loss_meta_gru_net, optimizer, data_loader, privacy_engine

if __name__ == "__main__":
    # set argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_number', type=int)
    parser.add_argument('--eval_interval', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--training_data_name', type=str)
    parser.add_argument('--network_type', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--meta_n_iter', type=int)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--accountant_mode', type=str)
    parser.add_argument('--meta_network_load_path', type=str)
    parser.add_argument('--transition_type', type=str)
    parser.add_argument('--meta_dist', type=str)
    parser.add_argument('--activate', type=str)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--dp_delta', type=float)
    parser.add_argument('--noise_multiplier', type=float)
    parser.add_argument('--clipping_bound', type=float)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--n_split', type=int)
    parser.add_argument('--is_dp', action='store_true')
    parser.add_argument('--train_all_layers', action='store_true')
    parser.add_argument('--post_process', action='store_true')
    parser.add_argument('--remove_first_value', action='store_true')
    parser.add_argument('--real_start', action='store_true')
    parser.add_argument('--eval_initial', action='store_true')
    parser.add_argument('--evaluate_first_next_location', action='store_true')
    parser.add_argument('--evaluate_second_next_location', action='store_true')
    parser.add_argument('--evaluate_global', action='store_true')
    parser.add_argument('--evaluate_source', action='store_true')
    parser.add_argument('--evaluate_target', action='store_true')
    parser.add_argument('--evaluate_route', action='store_true')
    parser.add_argument('--evaluate_destination', action='store_true')
    parser.add_argument('--evaluate_distance', action='store_true')
    parser.add_argument('--evaluate_empirical_next_location', action='store_true')
    parser.add_argument('--max_size', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--physical_batch_size', type=int)
    parser.add_argument('--coef_location', type=float)
    parser.add_argument('--coef_time', type=float)
    parser.add_argument('--n_pre_epochs', type=int)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--global_clip', type=int)
    parser.add_argument('--location_embedding_dim', type=int)
    parser.add_argument('--memory_dim', type=int)
    parser.add_argument('--memory_hidden_dim', type=int)
    parser.add_argument('--n_test_locations', type=int)
    parser.add_argument('--meta_patience', type=int)
    parser.add_argument('--privtree_theta', type=float)
    parser.add_argument('--clustering', type=str)
    args = parser.parse_args()
    
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.deterministic = True
    
    data_dir = get_datadir()
    data_path = data_dir / args.dataset / args.data_name / args.training_data_name
    save_path = data_dir / "results" / args.dataset / args.data_name / args.training_data_name / args.save_name
    save_path.mkdir(exist_ok=True, parents=True)
    (save_path / "imgs").mkdir(exist_ok=True, parents=True)

    # set logger
    logger = set_logger(__name__, save_path / "log.log")
    logger.info('log is saved to {}'.format(save_path / "log.log"))
    logger.info(f'used parameters {vars(args)}')

    # load dataset config    
    with open(data_path / "params.json", "r") as f:
        param = json.load(f)
    n_locations = param["n_locations"]
    max_time = param["max_time"]

    trajectories = load(data_path / "training_data.csv")
    time_trajectories = load(data_path / "training_data_time.csv")
    logger.info(f"load training data from {data_path / 'training_data.csv'}")
    logger.info(f"load time data from {data_path / 'training_data_time.csv'}")

    dataset = TrajectoryDataset(trajectories, time_trajectories, n_locations, args.n_split)
    dataset.compute_auxiliary_information(save_path, logger)
    param["format_to_label"] = dataset.format_to_label
    param["label_to_format"] = dataset.label_to_format

    data_loader = torch.utils.data.DataLoader(dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=args.batch_size, collate_fn=dataset.make_padded_collate(args.remove_first_value))
    logger.info(f"len of the dataset: {len(dataset)}")
    
    if args.batch_size == 0:
        args.batch_size = int(np.sqrt(len(dataset)))
        logger.info("batch size is set as " + str(args.batch_size))

    generator, eval_generator, loss_model, optimizer, data_loader, privacy_engine = construct_generator(data_loader)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=save_path / "checkpoint.pt", trace_func=logger.info)
    logger.info(f"early stopping patience: {args.patience}, save path: {save_path / 'checkpoint.pt'}")

    resultss = []
    epsilon = 0
    # traning
    generator.train()
    for epoch in tqdm.tqdm(range(args.n_epochs+1)):

        # evaluate the generator per eval_interval epochs 
        evaluate(epoch)       
            
        # when n_epochs is 0, only evaluate
        if args.n_epochs == 0:
            break

        # training
        if not args.is_dp:
            losses = train_epoch(data_loader, generator, optimizer)
        else:
            with BatchMemoryManager(data_loader=data_loader, max_physical_batch_size=min([args.physical_batch_size, args.batch_size]), optimizer=optimizer) as new_data_loader:
                losses = train_epoch(new_data_loader, generator, optimizer)
            epsilon = privacy_engine.get_epsilon(args.dp_delta)

        # early stopping
        early_stopping(np.sum(losses[:-1]), eval_generator)
        logger.info(f'epoch: {early_stopping.epoch} epsilon: {epsilon} | best loss: {early_stopping.best_score} | current loss: location {losses[:-2]}, time {losses[-2]}, norm {losses[-1]}')
        if early_stopping.early_stop:
            break

    try:
        logger.info(f"save model to {save_path / 'model.pt'}")
        torch.save(generator.state_dict(), save_path / "model.pt")
    except:
        logger.info("failed to save model because it is Markov1?")

    # concat vars(args) and param
    param.update(vars(args))
    logger.info(f"save param to {save_path / 'params.json'}")
    # logger.info(f"args: {param}")
    with open(save_path / "params.json", "w") as f:
        json.dump(param, f)