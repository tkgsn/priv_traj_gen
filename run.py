import argparse
import random
import numpy as np
import torch
import tqdm
from torch import nn, optim
import json
from scipy.spatial.distance import jensenshannon

from my_utils import get_datadir, load_dataset, load_time_dataset, clustering, privtree_clustering, noise_normalize, add_noise, plot_density, make_trajectories, set_logger
from dataset import TrajectoryDataset
from models import GRUNet, MetaGRUNet, MetaNetwork, MetaAttentionNetwork, MetaClassNetwork
from data_pre_processing import save_state_with_nan_padding
import torch.nn.functional as F
from opacus.utils.batch_memory_manager import BatchMemoryManager

from opacus import PrivacyEngine
from pytorchtools import EarlyStopping

from evaluation import count_source_locations, count_target_locations, compute_distribution_from_count, compute_route_count, compute_distance_distribution, compute_destination_count, compute_next_location_distribution, compute_global_distribution
from data_post_processing import post_process_chengdu

def evaluate():
    results = {}
    jss, generated_next_location_distributions = evaluate_next_location_on_test_dataset(first_next_location_distributions, test_data_loader, eval_generator)
    results["first_next_location_js"] = jss

    start_trajs = torch.tensor([traj[0] for traj in trajectories]).cuda(args.cuda_number)
    start_time_trajs = torch.tensor([time_traj[0] for time_traj in dataset.time_label_trajs]).cuda(args.cuda_number)
    start_data = [start_trajs, start_time_trajs] if args.real_start else None
    # generated_trajs, generated_time_trajs = make_sample(args.batch_size, generator, torch.tensor(dataset.labels), dataset.label_to_format, dataset.n_locations, TrajectoryDataset.time_end_idx(args.n_split), real_start=start_data, without_time=False)
    generated_trajs, generated_time_trajs = eval_generator.make_sample(dataset.references, dataset.n_locations, dataset._time_end_idx(), args.batch_size, real_start=start_data)
    generated_time_trajs = dataset.convert_time_label_trajs_to_time_trajs(generated_time_trajs)
    if (args.dataset == "chengdu") and args.post_process:
        generated_trajs = post_process_chengdu(generated_trajs)


    gene_global_distributions = [compute_global_distribution(generated_trajs, generated_time_trajs, time, dataset.n_locations, dataset._time_to_label) for time in [i[0] for i in dataset.time_ranges]]
    global_jss = []
    for i in range(len(dataset.time_ranges)):
        if gene_global_distributions[i] is None or global_distributions[i] is None:
            global_jss.append(1)
        else:
            js = jensenshannon(gene_global_distributions[i], global_distributions[i])**2
            global_jss.append(js)
    results["global_js"] = global_jss

    count = count_source_locations(trajectories)
    original_source_distribution = compute_distribution_from_count(count, dataset.n_locations)
    count = count_source_locations(generated_trajs)
    generated_source_distribution = compute_distribution_from_count(count, dataset.n_locations)
    source_js = jensenshannon(original_source_distribution, generated_source_distribution)**2
    results["source_js"] = source_js

    count = count_target_locations(trajectories)
    original_target_distribution = compute_distribution_from_count(count, dataset.n_locations)
    count = count_target_locations(generated_trajs)
    generated_target_distribution = compute_distribution_from_count(count, dataset.n_locations)
    target_js = jensenshannon(original_target_distribution, generated_target_distribution)**2
    results["target_js"] = target_js

    route_jss = []
    destination_jss = []
    next_location_jss = []
    for location in top_base_locations[:args.n_test_locations]:
        count = compute_route_count(trajectories, location)
        original_route_distribution = compute_distribution_from_count(count, dataset.n_locations)
        count = compute_route_count(generated_trajs, location)
        generated_route_distribution = compute_distribution_from_count(count, dataset.n_locations)
        route_js = jensenshannon(original_route_distribution, generated_route_distribution)**2
        route_jss.append(route_js)

        count = compute_destination_count(trajectories, location)
        original_destination_distribution = compute_distribution_from_count(count, dataset.n_locations)
        count = compute_destination_count(generated_trajs, location)
        generated_destination_distribution = compute_distribution_from_count(count, dataset.n_locations)
        destination_js = jensenshannon(original_destination_distribution, generated_destination_distribution)**2
        destination_jss.append(destination_js)

        generated_next_location_distribution = compute_next_location_distribution(location, generated_trajs, dataset.n_locations)
        if generated_next_location_distribution is None or next_location_distributions[location] is None:
            next_location_js = 1
        else:
            next_location_js = jensenshannon(next_location_distributions[location], generated_next_location_distribution)**2
            next_location_jss.append(next_location_js)
            plot_density(generated_next_location_distribution, dataset.n_locations, save_path / f"generated_empirical_next_location_distribution_{location}.png")
    
    results["empirical_next_location_js"] = next_location_jss
    results["route_js"] = route_jss
    results["destination_js"] = destination_jss

    n_bins = 100
    original_distance_distribution = compute_distance_distribution(distance_matrix, trajectories, n_bins)
    generated_distance_distribution = compute_distance_distribution(distance_matrix, generated_trajs, n_bins)
    distance_js = jensenshannon(original_distance_distribution, generated_distance_distribution)**2
    results["distance_js"] = distance_js

    return generated_next_location_distributions, results


def evaluate_next_location_on_test_dataset(next_location_distributions, data_loader, generator):
    cuda_number = next(generator.parameters()).device.index
    jss = []
    generated_next_location_distributions = {}
    for mini_batch in data_loader:
        input_locations = mini_batch["input"].cuda(cuda_number, non_blocking=True)
        references = [tuple(v) for v in mini_batch["reference"]]
        input_times = mini_batch["time"].cuda(cuda_number, non_blocking=True)
        output = torch.exp(generator([input_locations, input_times], references)[0]).cpu().detach().numpy()
        for i, traj in enumerate(input_locations):
            location = traj[1].item()
            next_location_distribution = next_location_distributions[location]
            if next_location_distribution is None:
                jss.append(0)
                print(f"WARNING: this base location {location} does not appear in the training set.")
            else:
                generated_next_location_distribution = output[i][1]
                # logger.info(f"generated_next_location_distribution: {generated_next_location_distribution}")
                jss.append(jensenshannon(next_location_distribution, generated_next_location_distribution)**2)
                plot_density(generated_next_location_distribution, len(next_location_distribution), save_path / f"first_next_location_distribution_{location}.png")
            
            generated_next_location_distributions[location] = generated_next_location_distribution.tolist()
                # print(i, min([jensenshannon(next_location_distribution, numpy_target_next_location_distributions[j])**2 for j in range(n_classes)]))
            # print(f"Jensen-Shannon divergence for location {location}: {jss}")
            # plot_density(generated_next_location_distribution, len(next_location_distribution), save_path / f"test_next_location_distribution_{location}.png")
    return jss, generated_next_location_distributions


def train_meta_network(meta_network, next_location_distributions, n_iter, early_stopping):
    optimizer = optim.Adam(meta_network.parameters(), lr=0.001)
    n_classes = len(next_location_distributions)
    with tqdm.tqdm(range(n_iter)) as pbar:
        for _ in pbar:
            losses = 0
            for i in range(n_classes):
                distribution = next_location_distributions[i]
                optimizer.zero_grad()

                # one_hot_encoding i
                i = F.one_hot(torch.tensor(i), num_classes=n_classes).to(next(iter(meta_network.parameters())).device).float().reshape(1,-1)
                meta_network_output = meta_network(i)
                loss = F.kl_div(meta_network_output, distribution, reduction='sum')
                loss.backward()
                optimizer.step()
                losses += loss.item() / n_classes
            pbar.set_description(f"loss: {losses:.5f}")
            early_stopping(losses, meta_network)

            if early_stopping.early_stop:
                break



def train_with_discrete_time(generator, optimizer, loss_model, input_locations, target_locations, input_times, target_times, labels, coef_location, coef_time):
    is_dp = hasattr(generator, "module")
    
    output_locations, output_times = generator([input_locations, input_times], labels)
    output_locations_v = output_locations.view(-1,output_locations.shape[-1])
    output_times_v = output_times.view(-1,output_times.shape[-1])
    # print(output_locations_v.shape)
    loss_location = loss_model(output_locations_v, target_locations.view(-1)) * coef_location
    loss_time = loss_model(output_times_v, target_times.view(-1)) * coef_time
    loss = loss_location + loss_time
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
        norms = [0]
    # norms = [0]

    optimizer.step()

    return loss_location.item(), loss_time.item(), norms


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
        elif dataset_name == "taxi":
            test_traj = np.concatenate([np.array(top_base_locations[:n_test_location]).reshape(-1,1), pad], axis=1).tolist()
            test_traj_time = [[0, 1, 2]]*n_test_location
        else:
            test_traj = np.concatenate([np.array(top_base_locations[:n_test_location]).reshape(-1,1), pad, np.array(top_base_locations[:n_test_location]).reshape(-1,1)], axis=1).tolist()
            test_traj_time = [[0, 800, 1200, 1439]]*n_test_location

    return test_traj, test_traj_time

def train_epoch(data_loader, generator, optimizer):
    loss_location = 0
    loss_time = 0
    norm = 0
    for i, batch in enumerate(data_loader):
        input_locations = batch["input"].cuda(args.cuda_number, non_blocking=True)
        target_locations = batch["target"].cuda(args.cuda_number, non_blocking=True)
        references = [tuple(v) for v in batch["reference"]]
        input_times = batch["time"].cuda(args.cuda_number, non_blocking=True)
        target_times = batch["time_target"].cuda(args.cuda_number, non_blocking=True)

        loss1, loss2, norms = train_with_discrete_time(generator, optimizer, loss_model, input_locations, target_locations, input_times, target_times, references, args.coef_location, args.coef_time)
        loss_location = loss_location + loss1
        loss_time = loss_time + loss2
        norm = norm + np.mean(norms)
    return loss_location / (i+1), loss_time / (i+1), norm / (i+1)

def construct_generator():
    if args.meta_network:
        target_next_location_distributions = []
        numpy_target_next_location_distributions = []

        # clustering
        if args.privtree_clustering:
            location_to_class = privtree_clustering(global_counts[0])
        else:
            location_to_class = clustering(noisy_global_distributions[0], distance_matrix, args.n_classes)
        args.n_classes = len(set(location_to_class.values()))
        
        for i in range(args.n_classes):
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
                target_distribution = noise_normalize(add_noise(next_location_distribution_i, sensitivity, args.epsilon))
                # compute js
                js = jensenshannon(real_target_distribution, target_distribution)**2
                logger.info(f"js divergence for class {i}: {js}")

            next_location_distribution_i = torch.tensor(target_distribution)
            
            target_next_location_distributions.append(next_location_distribution_i)
            numpy_target_next_location_distributions.append(next_location_distribution_i)

            plot_density(next_location_distribution_i, dataset.n_locations, save_path / f"class_next_location_distribution_{i}.png")

        for i in range(args.n_split):
            target_next_location_distributions.append(noisy_global_distributions[i])
        target_next_location_distributions = torch.stack(target_next_location_distributions).cuda(args.cuda_number)

        if args.meta_class:
            if args.attention:
                meta_network = MetaAttentionNetwork(args.hidden_dim, args.meta_hidden_dim, output_dim, len(target_next_location_distributions)).cuda(args.cuda_number)
            else:
                meta_network = MetaClassNetwork(args.hidden_dim, args.meta_hidden_dim, output_dim, len(target_next_location_distributions)).cuda(args.cuda_number)
        else:
            meta_network = MetaNetwork(args.n_classes, args.meta_hidden_dim, output_dim).cuda(args.cuda_number)
            args.hidden_dim = 0
        early_stopping = EarlyStopping(patience=args.meta_patience, path=save_path / "meta_network.pt")
        # set the meta network to not require gradients
        for name, param in meta_network.named_parameters():
            param.requires_grad = not args.fix_meta_network
            if name in ["embeddings.weight", "embeddings.bias"] and args.fix_embedding:
                param.requires_grad = not args.fix_embedding
        train_meta_network(meta_network, target_next_location_distributions, args.meta_n_iter, early_stopping)
        # meta_network = MetaAttentionNetwork(target_next_location_distributions)
        generator = MetaGRUNet(meta_network, input_dim, traj_type_dim, args.hidden_dim, output_dim, args.n_split+3, args.n_layers, args.embed_dim, dataset.reference_to_label).cuda(args.cuda_number)
        # generator = MetaGRUNet(meta_network, input_dim, traj_type_dim, args.hidden_dim, output_dim, args.n_split+3, args.n_layers, args.embed_dim, args.fix_meta_network, args.fix_embedding).cuda(args.cuda_number)
    else:
        generator = GRUNet(input_dim, traj_type_dim, args.hidden_dim, output_dim, args.n_split+3, args.n_layers, args.embed_dim, dataset.reference_to_label).cuda(args.cuda_number)
    return generator

def pretrain():
    if args.n_pre_epochs > 0:
        # noisy_global_distributions = torch.tensor(noisy_global_distributions).cuda(args.cuda_number)
        pre_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate)
        transition_matrix = [numpy_target_next_location_distributions[location_to_class[location]] for location in range(dataset.n_locations)]
        logger.info("make trajectories for pretraining")
        pre_trajs, pre_time_trajs = make_trajectories(noisy_global_distributions[0], reference_distribution, transition_matrix, time_distribution, len(dataset))
        for i in range(len(pre_time_trajs)):
            for j in range(len(pre_time_trajs[i])):
                pre_time_trajs[i][j] = TrajectoryDataset.label_to_time(pre_time_trajs[i][j], args.n_split, max_time)
        pre_dataset = TrajectoryDataset(pre_trajs, pre_time_trajs, n_bins, args.n_split, max_time)
        pre_dataset.n_locations = dataset.n_locations
        pretrain_batch = 100
        pre_data_loader = torch.utils.data.DataLoader(pre_dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=pretrain_batch, collate_fn=make_padded_collate(dataset.n_locations, dataset.format_to_label, dataset._time_to_label, remove_first_value=False))
        # if hasattr(generator, "meta_network"):
        #     generator.set_requires_grad_of_meta_network(False)
        early_stopping = EarlyStopping(patience=10, verbose=True, path=save_path / "pretrained_network.pt")
        for k in tqdm.tqdm(range(args.n_pre_epochs)):
            loss_location, loss_time, norm = train_epoch(pre_data_loader, generator, pre_optimizer)
            early_stopping((loss_location + loss_time), generator)
            if k % 10 == 0:
                logger.info(f"pretrain epoch {k}: loss_location {loss_location}, loss_time {loss_time}, norm {norm}")
            if early_stopping.early_stop:
                break

if __name__ == "__main__":
    # set argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_number', type=int)
    parser.add_argument('--eval_interval', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--training_data_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--meta_n_iter', type=int)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--accountant_mode', type=str)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--noise_multiplier', type=float)
    parser.add_argument('--clipping_bound', type=float)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--n_split', type=int)
    parser.add_argument('--is_dp', action='store_true')
    parser.add_argument('--post_process', action='store_true')
    parser.add_argument('--remove_first_value', action='store_true')
    parser.add_argument('--meta_network', action='store_true')
    parser.add_argument('--fix_meta_network', action='store_true')
    parser.add_argument('--meta_class', action='store_true')
    parser.add_argument('--fix_embedding', action='store_true')
    parser.add_argument('--real_start', action='store_true')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--privtree_clustering', action='store_true')
    parser.add_argument('--max_size', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--physical_batch_size', type=int)
    parser.add_argument('--coef_location', type=float)
    parser.add_argument('--coef_time', type=float)
    parser.add_argument('--n_pre_epochs', type=int)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--global_clip', type=int)
    parser.add_argument('--meta_hidden_dim', type=int)
    parser.add_argument('--n_test_locations', type=int)
    parser.add_argument('--meta_patience', type=int)
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

    # set logger
    logger = set_logger(__name__, save_path / "log.log")
    logger.info('log is saved to {}'.format(save_path / "log.log"))
    logger.info(f'used parameters {vars(args)}')

    # load dataset config    
    with open(data_path / "params.json", "r") as f:
        param = json.load(f)
    n_locations = param["n_locations"]
    max_time = param["max_time"]

    training_data_path = data_path / "training_data.csv"
    trajectories = load_dataset(training_data_path, logger=logger)
    time_trajectories = load_time_dataset(data_path, logger=logger)
    dataset = TrajectoryDataset(trajectories, time_trajectories, n_locations, args.n_split, max_time)
    logger.info(f"len of the dataset: {len(dataset)}")
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=args.batch_size, collate_fn=dataset.make_padded_collate(args.remove_first_value))

    param["format_to_label"] = dataset.format_to_label
    param["label_to_format"] = dataset.label_to_format
    
    if args.batch_size == 0:
        args.batch_size = int(np.sqrt(len(dataset)))
        logger.info("batch size is set as " + str(args.batch_size))

    # input_dim := n_locations + start and end
    input_dim = dataset.n_locations+2
    output_dim = dataset.n_locations
    traj_type_dim = len(dataset.label_to_reference)
    n_split = args.n_split
    sensitivity = args.global_clip+1

    
    # clipped_trajectories = global_clipping(trajectories, args.global_clip)
    # clipped_trajectories = trajectories
    top_k_locations, next_location_counts, first_next_location_counts, global_counts, label_count, time_distribution, reference_distribution = dataset.compute_auxiliary_information(save_path, logger)
    next_location_distributions = {key: noise_normalize(next_location_count) for key, next_location_count in next_location_counts.items()}
    first_next_location_distributions = {key: noise_normalize(first_next_location_count) for key, first_next_location_count in first_next_location_counts.items()}
    global_distributions = [noise_normalize(global_count) for global_count in global_counts]
    noisy_global_distributions = [noise_normalize(add_noise(global_count, sensitivity, args.epsilon)) for global_count in global_counts]
    for i in range(len(noisy_global_distributions)):
        if noisy_global_distributions[i] == None:
            noisy_global_distributions[i] = [1/dataset.n_locations] * dataset.n_locations
    noisy_global_distributions = torch.tensor(noisy_global_distributions)
    top_base_locations = np.argsort(global_counts[0])[::-1]
    param["test_locations"] = top_base_locations[:args.n_test_locations].tolist()
    test_traj, test_traj_time = make_test_data(data_path, top_base_locations, args.n_test_locations, args.dataset)
    test_dataset = TrajectoryDataset(test_traj, test_traj_time, n_locations, args.n_split, max_time)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, shuffle=False, pin_memory=True, batch_size=args.batch_size, collate_fn=test_dataset.make_padded_collate(args.remove_first_value))


    epsilon_for_sgd = float("inf")
    param["global_distributions"] = noisy_global_distributions.tolist()

    distance_matrix = np.load(data_path / "distance_matrix.npy")
    generator = construct_generator()
    loss_model = nn.NLLLoss(ignore_index=dataset.ignore_idx(dataset.n_locations))
    pretrain()

    optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate)
    if args.is_dp:
        privacy_engine = PrivacyEngine(accountant=args.accountant_mode)
        generator, optimizer, data_loader = privacy_engine.make_private(module=generator, optimizer=optimizer, data_loader=data_loader, noise_multiplier=args.noise_multiplier, max_grad_norm=args.clipping_bound)
        eval_generator = generator._module
        delta = 1e-5
    else:
        eval_generator = generator
 
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=save_path / "checkpoint.pt", trace_func=logger.info)
    logger.info(f"early stopping patience: {args.patience}, save path: {save_path / 'checkpoint.pt'}")
    

    resultss = []
    # traning
    generator.train()
    for epoch in tqdm.tqdm(range(args.n_epochs+1)):

        if epoch % args.eval_interval == 0:
            generator.eval()
            with torch.no_grad():
                generated_next_location_distributions, results = evaluate()
            logger.info(f"evaluation: {results}")
            generator.train()
            resultss.append(results)

            args.end_epoch = early_stopping.epoch
            args.end_loss = early_stopping.best_score
            args.results = resultss

            param.update(vars(args))
            logger.info(f"save param to {save_path / 'params.json'}")
            with open(save_path / "params.json", "w") as f:
                json.dump(param, f)

        lossess = []

        if not args.is_dp:
            loss_location, loss_time, norm = train_epoch(data_loader, generator, optimizer)
        else:
            phisical_batch_size = min([args.physical_batch_size, args.batch_size])
            with BatchMemoryManager(data_loader=data_loader, max_physical_batch_size=phisical_batch_size, optimizer=optimizer) as new_data_loader:
                loss_location, loss_time, norm = train_epoch(new_data_loader, generator, optimizer)

            try:
                epsilon = privacy_engine.get_epsilon(delta)
            except:
                print("error!")
                epsilon = 0
            logger.info(f'epsilon{epsilon}/{epsilon_for_sgd}, delta{delta}')

        early_stopping((loss_location + loss_time), eval_generator)
        logger.info(f'epoch: {early_stopping.epoch} | best loss: {early_stopping.best_score} | current loss: (location {loss_location} + time {loss_time}) | norm: {norm}')

        if early_stopping.early_stop:
            break


    param["generated_next_location_distributions"] = generated_next_location_distributions


    start_trajs = torch.tensor([traj[0] for traj in trajectories]).cuda(args.cuda_number)
    start_time_trajs = torch.tensor([time_traj[0] for time_traj in dataset.time_label_trajs]).cuda(args.cuda_number)
    start_data = [start_trajs, start_time_trajs] if args.real_start else None
    generated_trajs, generated_time_trajs = eval_generator.make_sample(dataset.references, dataset.n_locations, dataset._time_end_idx(), args.batch_size, real_start=start_data)
    generated_data_path = save_path / f"generated_trajs.csv"
    generated_time_data_path = save_path / f"genenerated_time_trajs.csv"
    save_state_with_nan_padding(generated_data_path, generated_trajs)
    save_state_with_nan_padding(generated_time_data_path, generated_time_trajs)
    logger.info(f"save generated data to {generated_data_path}")

    torch.save(generator.state_dict(), save_path / "model.pt")

    # concat vars(args) and param
    param.update(vars(args))
    logger.info(f"save param to {save_path / 'params.json'}")
    # logger.info(f"args: {param}")
    with open(save_path / "params.json", "w") as f:
        json.dump(param, f)