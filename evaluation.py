import torch
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

from my_utils import construct_default_quadtree
from collections import Counter
import numpy as np

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

def evaluate_next_location_on_test_dataset(next_location_distributions, data_loader, generator, target_index, order=1):
    jss = []
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
            output = torch.exp(output).cpu().detach().numpy()[:, target_index]
        # catch the key error
        try:
            if order == 1:
                targets = torch.tensor([next_location_distributions[traj[target_index].item()] for traj in input_locations])
            elif order == 2:
                targets = torch.tensor([next_location_distributions[(traj[target_index-1].item(), traj[target_index].item())].tolist() for traj in input_locations])
        except KeyError:
            continue
        jss.append(compute_distribution_js_for_each_depth(output, targets))
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

def count_passing_locations(trajectories):
    # count the appearance of locations
    passing_locations = sum(trajectories, [])
    return Counter(passing_locations)

def count_target_locations(trajs):
    target_locations = []
    for traj in trajs:
        target_locations.append(traj[-1])
    return Counter(target_locations)

def compute_distribution_from_count(count, n_locations):
    distribution = np.zeros(n_locations)
    for key, value in count.items():
        distribution[key] = value
    distribution = distribution / np.sum(distribution)
    return distribution

# compute the route distribution
# i.e., given the source location, compute the probability of each location passing through
def compute_route_count(trajs, source_location):
    # find the trajs that start from the source location
    trajs_from_source = [traj for traj in trajs if traj[0] == source_location]
    # compute the route distribution
    route_locations = []
    for traj in trajs_from_source:
        route_locations.extend(traj[1:])
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



def compute_distance_distribution(distance_matrix, trajs, n_bins):
    distances = compute_distances(distance_matrix, trajs)
    # make histogram using n_bins
    hist, bin_edges = np.histogram(distances, bins=n_bins)
    # compute prob
    distribution = hist / np.sum(hist)
    return distribution


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
# ex) traj [3,24,25,3], time_traj [0,3,3,4,5]
# at time 0 -> 3
# at time 3 -> 24,25
# at time 4 -> 3
def compute_global_counts_from_time_label(trajs, time_label_trajs, time_label, n_locations):
    # find the locations at time
    def locations_at_time(traj, time_label_traj, time):
        indice = [i for i, t in enumerate(time_label_traj) if t == time]
        return [traj[i] for i in indice]
    
    locations = []
    for traj, time_label_traj in zip(trajs, time_label_trajs):
        locations.extend(locations_at_time(traj, time_label_traj, time_label))
    location_count = Counter({location:0 for location in range(n_locations)})
    location_count.update(locations)
    location_count = [count for _, count in location_count.items()]

    return location_count



def compute_global_counts(trajectories, real_time_traj, time, n_locations, time_to_label):
    def location_at_time(trajectory, time_traj, t):
        if t == 0:
            return int(trajectory[0])

        label = time_to_label(t)
        time_label_traj = [time_to_label(time) for time in time_traj]
        if label not in time_label_traj:
            return None
        elif time_label_traj.index(label) == len(trajectory):
            return None
        else:
            return trajectory[time_label_traj.index(label)]
            
    locations = []
    count = 0
    for trajectory, time_traj in zip(trajectories, real_time_traj):
        if 1+len(trajectory) != len(time_traj):
            # print("BUG, NEED TO BE FIXED", trajectory, time_traj)
            count += 1
        else:
            locations.append(location_at_time(trajectory, time_traj, time))

    # count each location and conver to probability
    location_count = {i:0 for i in range(n_locations)}
    for location in locations:
        if location is not None:
            location_count[location] += 1
    location_counts = {location: count for location, count in location_count.items()}
    return list(location_counts.values())

def compute_global_distribution(trajectories, real_time_traj, time, n_locations, time_to_label):
    global_counts = compute_global_counts(trajectories, real_time_traj, time, n_locations, time_to_label)
    global_distribution = []
    summation = sum(global_counts)
    if summation == 0:
        return None
    for count in global_counts:
        global_distribution.append(count / sum(global_counts))
    return global_distribution