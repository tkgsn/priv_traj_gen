import torch
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
import numpy as np

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

def compute_next_location_count(target, trajectories, n_locations, next_first=False):
    # compute the count of next location for each location
    if next_first:
        count = Counter([trajectory[1] for trajectory in trajectories if trajectory[0]==target and len(trajectory)>1])
    else:
        count = Counter([trajectory[i+1] for trajectory in trajectories for i in range(len(trajectory)-1) if trajectory[i]==int(target)])
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
        indice = [i for i, t in enumerate(time_label_traj[:-1]) if t == time]
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