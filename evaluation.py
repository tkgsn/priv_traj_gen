import torch
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

from my_utils import construct_default_quadtree, noise_normalize, save, plot_density
from collections import Counter
import numpy as np
import scipy
import random
import pathlib
import sqlite3



def run(generator, dataset, args, epoch):

    n_bins = int(np.sqrt(dataset.n_locations)-2)

    if args.compensation:
        route_db_path = f"/data/{args.dataset}/pair_to_route/{n_bins}/paths.db"
        print("compensating trajectories by", route_db_path)
    else:
        print("not compensating trajectories")

    if epoch % args.eval_interval != 0:
        return
    if epoch == 0 and not args.eval_initial:
        return

    n_test_locations = min(args.n_test_locations, len(dataset.top_base_locations))

    generator.eval()
    with torch.no_grad():
        results = {}

        if args.evaluate_first_next_location:
            jss = evaluate_next_location_on_test_dataset(dataset.first_next_location_counts, dataset.first_order_test_data_loader, dataset.first_counters, generator, 1)
            results["first_next_location_js"] = jss

        # if args.evaluate_second_next_location and (dataset.seq_len > 2):
        #     jss = evaluate_next_location_on_test_dataset(dataset.second_next_location_counts, dataset.second_order_test_data_loader, dataset.second_counters, generator, 2)
        #     results["second_next_location_js"] = jss

        if args.evaluate_second_order_next_location and (dataset.seq_len > 2):
            jss = evaluate_next_location_on_test_dataset(dataset.second_order_next_location_counts, dataset.second_order_test_data_loader, dataset.second_counters, generator, 2)
            results["second_order_next_location_js"] = jss

        if (args.evaluate_global or args.evaluate_passing or args.evaluate_source or args.evaluate_target or args.evaluate_route or args.evaluate_destination or args.evaluate_distance):

            counters = {"global":[Counter() for _ in dataset.time_ranges], "passing": Counter(), "source": Counter(), "target": [Counter() for _ in range(n_test_locations)], "route": [Counter() for _ in range(n_test_locations)], "destination": [Counter() for _ in range(n_test_locations)], "distance": Counter(), "first_location": Counter()}
            condition = True
            counter = 0
            gene_trajs = []
            print("generating...")
            while condition:
                counter += 1
                mini_batch_size =  min([100, len(dataset.references)])
                # sample mini_batch_size references from dataset.references
                references = random.sample(dataset.references, mini_batch_size)
                generated = generator.make_sample(references, mini_batch_size)

                if len(generated) == 2:
                    # when n_data == 2 it causes bug
                    generated_trajs, generated_time_trajs = generated
                    # generated_time_trajs = dataset.convert_time_label_trajs_to_time_trajs(generated_time_trajs)
                else:
                    generated_trajs = generated
                    generated_time_trajs = dataset.time_label_trajs

                if args.evaluate_passing or args.evaluate_route:
                    if args.compensation:
                        compensated_trajs = compensate_trajs(generated_trajs, route_db_path)
                    else:
                        compensated_trajs = generated_trajs

                first_locations = [traj[0] for traj in generated_trajs if len(traj) > 1]
                counters["first_location"] += Counter(first_locations)

                if args.evaluate_global:
                    for time_label in range(1, dataset.n_time_split+1):
                        counters["global"][time_label-1] += compute_global_counts_from_time_label(generated_trajs, generated_time_trajs, time_label)
                
                if args.evaluate_passing:
                    counters["passing"] += count_passing_locations(compensated_trajs)

                if args.evaluate_source:
                    counters["source"] += count_source_locations(generated_trajs)

                for i, location in enumerate(dataset.top_base_locations[:n_test_locations]):
                    if args.evaluate_target:
                        counters["target"][i] += count_target_locations(generated_trajs, location)

                    if args.evaluate_destination:
                        counters["destination"][i] += count_route_locations(generated_trajs, location)

                    if args.evaluate_route:
                        counters["route"][i] += count_route_locations(compensated_trajs, location)

                if args.evaluate_distance:
                    counters["distance"] += count_distance(dataset.distance_matrix, generated_trajs, dataset.n_bins_for_distance)
        
                # evaluate the same number of generated data as the ten times of that of original data
                condition = counter < len(dataset.references)*10 / mini_batch_size

                gene_trajs.extend(generated_trajs)

            # save

            save(pathlib.Path(args.save_path) / f"evaluated_{epoch}.csv", gene_trajs)
            print(f"saved evaluated file ({len(gene_trajs)}) to", pathlib.Path(args.save_path) / f"evaluated_{epoch}.csv")

            n_gene_traj = mini_batch_size * counter
            # compute js
            real_counters = dataset.real_counters
            n_trajs = dataset.n_trajs
            for key, counter in counters.items():
                print(key)
                if key == "first_location":
                    continue
                if key == "distance":
                    n_vocabs = dataset.n_bins_for_distance
                else:
                    n_vocabs = dataset.n_locations


                # compute kl divergence for a dimension
                if key in ["target", "destination", "route"]:
                    results[f"{key}_kls_eachdim"] = [compute_divergence(real_counter, n_traj, counter_, counters["first_location"][location], n_vocabs) for counter_, real_counter, n_traj, location in zip(counter, real_counters[key], n_trajs[key], dataset.top_base_locations)]
                    results[f"{key}_kls_positivedim"] = [compute_divergence(real_counter, n_traj, counter_, counters["first_location"][location], n_vocabs, positive=True) for counter_, real_counter, n_traj, location in zip(counter, real_counters[key], n_trajs[key], dataset.top_base_locations)]
                elif key == "global":
                    results[f"{key}_kls_eachdim"] = [compute_divergence(real_counter, n_traj, counter_, n_gene_traj, n_vocabs) for counter_, real_counter, n_traj in zip(counter, real_counters[key], n_trajs[key])]
                    results[f"{key}_kls_positivedim"] = [compute_divergence(real_counter, n_traj, counter_, n_gene_traj, n_vocabs, positive=True) for counter_, real_counter, n_traj in zip(counter, real_counters[key], n_trajs[key])]
                else:
                    results[f"{key}_kl_eachdim"] = compute_divergence(real_counters[key], n_trajs[key], counter, n_gene_traj, n_vocabs)
                    results[f"{key}_kl_positivedim"] = compute_divergence(real_counters[key], n_trajs[key], counter, n_gene_traj, n_vocabs, positive=True)

                # compute js divergence
                if key in ["target", "destination", "route"]:
                    results[f"{key}_jss"] = []
                    for i, (counter_, real_counter) in enumerate(zip(counter, real_counters[key])):
                        results[f"{key}_jss"].append(compute_divergence(real_counter, sum(real_counter.values()), counter_, sum(counter_.values()), n_vocabs, axis=1))
                        plot_density(counter_, dataset.n_locations, pathlib.Path(args.save_path) / "imgs" / f"{key}_{i}.png", dataset.top_base_locations[i])
                elif key == "global":
                    for i, (counter_, real_counter) in enumerate(zip(counter, real_counters[key])):
                        results[f"{key}_jss_{i}"] = compute_divergence(real_counter, sum(real_counter.values()), counter_, sum(counter_.values()), n_vocabs, axis=1)
                        plot_density(counter_, dataset.n_locations, pathlib.Path(args.save_path) / "imgs" / f"{key}_{i}.png")
                else:
                    results[f"{key}_js"] = compute_divergence(real_counters[key], sum(real_counters[key].values()), counter, sum(counter.values()), n_vocabs, axis=1)
                    plot_density(counter, n_vocabs, pathlib.Path(args.save_path) / "imgs" / f"{key}.png")
    generator.train()

    return results


def compensate_trajs(trajs, db_path):
    new_trajs = []
    counter = 0
    for traj in trajs:
        invalid_path = False
        if len(traj) == 1:
            new_trajs.append(traj)
        else:
            new_traj = [traj[0]]
            for i in range(len(traj)-1):
                edges = compensate_edge_by_map(traj[i], traj[i+1], db_path)
                invalid_path = invalid_path or (len(edges) == 0)
                new_traj.extend(edges[1:])
            if not invalid_path:
                new_trajs.append(new_traj)
            else:
                counter += 1
    print("WARNING: n invalid trajs", counter,  "/", len(trajs))
    return new_trajs

def compensate_edge_by_map(from_state, to_state, db_path):

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute(f"SELECT route FROM state_edge_to_route WHERE start_state={from_state} AND end_state={to_state}")
        edges = c.fetchone()
        if edges is None:
            # print("WARNING: path not exist", from_state, to_state)
            return []
        else:
            state_routes = eval(edges[0])
            # choose the shortest one
            shortest_route = min(state_routes, key=lambda x: len(x))
            return shortest_route



def compute_divergence(real_count, n_real_traj, inferred_count, n_gene_traj, n_vocabs, axis=0, positive=False):
    if n_real_traj == 0:
        print("WARNING: n_real_traj is zero")
        raise ValueError("no trajectory is evaluated")
    if n_gene_traj == 0:
        print("WARNING: n_gene_traj is zero, convert to uniform")
        inferred_count = {key: 1 for key in range(n_vocabs)}
        n_gene_traj = n_vocabs
    if axis == 0:

        # compute the kl divergence on the dimensions that are positive
        real_distribution = compute_distribution_from_count(real_count, n_vocabs, n_real_traj)
        real_distribution = np.stack([real_distribution, 1-real_distribution], axis=0)
        inferred_distribution = compute_distribution_from_count(inferred_count, n_vocabs, n_gene_traj)
        inferred_distribution = np.stack([inferred_distribution, 1-inferred_distribution], axis=0)
        # plus epsilon value to avoid inf for zero dimension
        inferred_distribution[inferred_distribution == 0] = 1e-10

        if positive:
            # filter out the negative dimensions
            positive_indices = np.where(real_distribution[0] > 0)[0]
            real_distribution = real_distribution[:, positive_indices]
            inferred_distribution = inferred_distribution[:, positive_indices]
            # print(positive_indices)
            # print([real_count[i] for i in positive_indices])
            # print([inferred_count[i] for i in positive_indices])

            a = scipy.stats.entropy(real_distribution, inferred_distribution, axis=0)
            # print(a)
            # print(np.argmax(a), positive_indices[np.argmax(a)], real_count[positive_indices[np.argmax(a)]], n_real_traj, inferred_count[positive_indices[np.argmax(a)]], n_gene_traj)
            # print(real_distribution[:, np.argmax(a)], inferred_distribution[:, np.argmax(a)])

        if scipy.stats.entropy(real_distribution, inferred_distribution, axis=0).sum() == float("inf"):
            for i in range(n_vocabs):
                if scipy.stats.entropy(real_distribution[:, i], inferred_distribution[:, i], axis=0) == float("inf"):
                    print(i)
                    print(real_distribution[:, i])
                    print(inferred_distribution[:, i])
                    if i in real_count:
                        print(real_count[i])
                    else:
                        print("not in real")
                    if i in inferred_count:
                        print(inferred_count[i])
                        print("not in inferred")
                    print(n_real_traj, real_count)
                    print(n_gene_traj, inferred_count)
                    raise ValueError("inf")
        
        return scipy.stats.entropy(real_distribution, inferred_distribution, axis=0).sum()
    else:
        # real_count and inferred_count will be the probability distributions
        assert n_real_traj == sum(real_count.values()), "n_real_traj must be equal to sum(real_count.values())"
        assert n_gene_traj == sum(inferred_count.values()), "n_gene_traj must be equal to sum(inferred_count.values())"
        real_distribution = compute_distribution_from_count(real_count, n_vocabs, n_real_traj)
        inferred_distribution = compute_distribution_from_count(inferred_count, n_vocabs, n_gene_traj)
        return jensenshannon(real_distribution, inferred_distribution)**2



def make_target_distributions_of_all_layers(target_distribution, tree):
    # from the location distribution on the all states (i.e., leafs), make the target distribution of all layers
    # target_distribution: (batch_size, n_locations)
    tree._register_count_to_complete_graph(target_distribution)
    distributions = [target_distribution]
    for depth in list(range(tree.max_depth))[1:][::-1]:
        nodes = tree.get_nodes(depth)
        for node in nodes:
            node.count = 0
            for child in node.children:
                node.count += child.count
        distribution = {node: node.count for node in nodes}
        # sort according to node.oned_coordinate
        distribution = torch.stack([v for _, v in sorted(distribution.items(), key=lambda item: item[0].oned_coordinate)], dim=1)
        distributions.append(distribution)
    return distributions[::-1]


def compute_distribution_js_for_each_depth(distribution, target_distribution):
    next_location_js_for_all_depth = []
    n_locations = distribution.shape[-1]
    tree = construct_default_quadtree(int(np.sqrt(n_locations))-2)
    tree.make_self_complete()
    target_next_location_distribution_for_all_depth = make_target_distributions_of_all_layers(torch.tensor(target_distribution).view(-1,n_locations), tree)
    generated_next_location_distribution_for_all_depth = make_target_distributions_of_all_layers(torch.tensor(distribution).view(-1,n_locations), tree)
    for depth in range(1, tree.max_depth+1):
        next_location_js_for_all_depth.append(jensenshannon(target_next_location_distribution_for_all_depth[depth-1], generated_next_location_distribution_for_all_depth[depth-1], axis=1)**2)
    return np.stack(next_location_js_for_all_depth, axis=1).tolist()


def evaluate_next_location_on_test_dataset(next_location_counts, data_loader, counters, generator, target_index):
    next_location_distributions = {key: noise_normalize(next_location_count) for key, next_location_count in next_location_counts.items()}
    jss = []

    outputs = []
    for mini_batch in data_loader:
        if hasattr(generator, "transition_matrix"):
            input_locations = mini_batch["input"]
            output = torch.exp(generator(input_locations[:, target_index]))
        else:
            device = next(iter(generator.parameters())).device
            input_locations = mini_batch["input"].to(device)
            references = [tuple(v) for v in mini_batch["reference"]]
            input_times = mini_batch["time"].to(device)
            output = generator([input_locations, input_times], references)[0]
            output = output[-1] if type(output) == list else output
            output = torch.exp(output).cpu().detach().numpy()[:, target_index].tolist()
            outputs.extend(output)
    
    cursor = 0
    for target, n_test_data in counters.items():
        output = outputs[cursor:cursor+n_test_data]
        inferred_distribution = np.mean(output, axis=0)
        target_distribution = next_location_distributions[target]
        jss.append(compute_distribution_js_for_each_depth(inferred_distribution, target_distribution))

        cursor += n_test_data

    return jss


def compute_destination_count(trajs, source_location):
    # find the trajs that start from the source location
    trajs_from_source = [traj for traj in trajs if traj[0] == source_location]
    # compute the route distribution
    route_locations = []
    for traj in trajs_from_source:
        route_locations.append(traj[-1])
    route_count = Counter(route_locations)
    return route_count

def count_source_locations(trajs):
    start_locations = []
    for traj in trajs:
        start_locations.append(traj[0])
    return Counter(start_locations)

def count_passing_locations(trajs):
    # count the appearance of locations
    passing_locations = []
    for traj in trajs:
        passing_locations.extend(list(set(traj[1:])))
    return Counter(passing_locations)

def count_target_locations(trajs, source_location):
    trajs = [traj for traj in trajs if traj[0] == source_location and len(traj) > 1]
    # trajs = [traj for traj in trajs if traj[0] == source_location]
    target_locations = []
    for traj in trajs:
        target_locations.append(traj[-1])
    return Counter(target_locations)

def compute_distribution_from_count(count, n_locations, n_trajs):
    distribution = np.zeros(n_locations)
    for key, value in count.items():
        distribution[key] = value
    return distribution / n_trajs

# compute the route distribution
# i.e., given the source location, compute the probability of each location passing through
def count_route_locations(trajs, source_location):
    # find the trajs that start from the source location
    trajs_from_source = [traj for traj in trajs if traj[0] == source_location]
    # compute the route distribution
    route_locations = []
    for traj in trajs_from_source:
        route_locations_ = list(set(traj[1:]) - set([source_location]))
        route_locations.extend(route_locations_)
    route_count = Counter(route_locations)
    return route_count

def compute_distance(distance_matrix, traj):
    distance = 0
    for i in range(len(traj)-1):
        distance += distance_matrix[traj[i], traj[i+1]]
    return distance

def compute_distances(distance_matrix, trajs):
    distances = []
    for traj in trajs:
        distance = 0
        for i in range(len(traj)-1):
            distance += distance_matrix[traj[i]][traj[i+1]]
        distances.append(distance)
    return distances



def count_distance(distance_matrix, trajs, n_bins):
    distances = compute_distances(distance_matrix, trajs)
    # make histogram using n_bins
    hist, _ = np.histogram(distances, bins=n_bins)
    # compute prob
    count = {key:v for key, v in enumerate(hist)}
    return count


def compute_next_location_distribution(target, trajectories, n_locations):
    # compute the next location probability for each location
    counts = compute_next_location_count(target, trajectories, n_locations)
    summation = sum(counts)
    if summation == 0:
        return None
    distribution = [count/summation for count in counts]
    return distribution

def compute_next_location_count(target, trajectories, n_locations, target_index=0):
    # compute the count of next location for each location
    if target_index == 0:
        count = Counter([trajectory[i+1] for trajectory in trajectories for i in range(len(trajectory)-1) if trajectory[i]==int(target)])
    elif target_index == 1:
        trajectories = [trajectory for trajectory in trajectories if len(trajectory)>1]
        count = Counter([trajectory[1] for trajectory in trajectories if trajectory[0]==target])
    elif target_index == 2:
        trajectories = [trajectory for trajectory in trajectories if len(trajectory)>2]
        count = Counter([trajectory[2] for trajectory in trajectories if trajectory[1]==target])
    counts = []
    for i in range(n_locations):
        if i not in count:
            counts.append(0)
        else:
            counts.append(count[i])
    return counts



# compute each count for each time split
# ex) traj [3,24,25,3], time_traj [1,3,3,4]
# at time 1~3 -> 3
# at time 3~4 -> 24,25
# at time 4~ -> 3
# note at time 2 -> 3
def compute_global_counts_from_time_label(trajs, time_label_trajs, time_label):
    # find the locations at time
    def locations_at_time(traj, time_label_traj, time_label):
        # if time_label in time_label_traj, return the locations at time
        # else, return the final location of the index that is the closest to the time_label
        assert time_label >= 1, "time_label should be larger than 1 because 0 is the start signal"
        if time_label in time_label_traj:
            indice = [i for i, t in enumerate(time_label_traj) if t == time_label]
            return [traj[i] for i in indice]
        else:
            indice = [i for i, t in enumerate(time_label_traj) if t <= time_label]
            if len(indice) == 0:
                return []
            else:
                return [[traj[i] for i in indice][-1]]

    locations = []
    for traj, time_label_traj in zip(trajs, time_label_trajs):
        locations.extend(locations_at_time(traj, time_label_traj, time_label))

    location_count = Counter(locations)

    return location_count


# def compute_global_counts(trajectories, real_time_traj, time, n_locations, time_to_label):
#     def location_at_time(trajectory, time_traj, t):
#         if t == 0:
#             return int(trajectory[0])

#         label = time_to_label(t)
#         time_label_traj = [time_to_label(time) for time in time_traj]
#         if label not in time_label_traj:
#             return None
#         elif time_label_traj.index(label) == len(trajectory):
#             return None
#         else:
#             return trajectory[time_label_traj.index(label)]
            
#     locations = []
#     count = 0
#     for trajectory, time_traj in zip(trajectories, real_time_traj):
#         if 1+len(trajectory) != len(time_traj):
#             # print("BUG, NEED TO BE FIXED", trajectory, time_traj)
#             count += 1
#         else:
#             location = location_at_time(trajectory, time_traj, time)
#             if location is not None:
#                 locations.append(location)

#     # count each location
#     location_count = Counter(locations)
#     return location_count

# def compute_global_distribution(trajectories, real_time_traj, time, n_locations, time_to_label):
#     global_counts = compute_global_counts(trajectories, real_time_traj, time, n_locations, time_to_label)
#     global_distribution = compute_distribution_from_count(global_counts, n_locations)
#     return global_distribution