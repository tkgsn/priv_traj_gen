import pandas as pd
import json
import numpy as np
import bisect
import random
from sklearn.preprocessing import normalize
import pathlib
import torch
from collections import Counter
from logging import getLogger, config
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

# make a dataset for pre-training
def make_trajectories(global_distribution, reference_distribution, transition_matrix, time_distribution, n_samples):
    seq_len = max([len(v) for v in time_distribution.keys()])
    length_to_candidates = {length-1: [reference for _, reference in enumerate(time_distribution.keys()) if len(reference) == length] for length in range(2, seq_len + 2)}

    def sample_time(length):
        # sample a time from time_distribution and length
        # candidates are the times with the same length as label
        candidates = length_to_candidates[length]
        distribution = [time_distribution[candidate] for candidate in candidates]
        distribution = [d/sum(distribution) for d in distribution]
        indice = np.random.choice(range(len(distribution)), size=1, p=distribution)
        label = candidates[indice[0]]
        return list(label)

    n_references = len(reference_distribution)
    all_references = list(reference_distribution.keys())
    # sample reference according to reference_distribution
    # reference_distribution: a dictionary of count of each reference (i.e., {reference: count})
    sum_ = sum(list(reference_distribution.values()))
    probs = [count/sum_ for count in reference_distribution.values()]
    indice = np.random.choice(n_references, size=n_samples, p=probs)
    references = [all_references[index] for index in indice]
    lengths = [len(reference) for reference in references]
    # sample time according to time_distribution
    times = [sample_time(length) for length in lengths]

    # make traj according to references
    # sample a start location from global_distribution
    # sample a next location from transition_matrix[previous_locaction]
    trajectories = []
    start_locations = np.random.choice(len(global_distribution), size=n_samples, p=global_distribution)
    for reference, start_location in tqdm.tqdm(zip(references, start_locations)):
        trajectory = [start_location]
        for i in range(len(reference)-1):
            previous_location = int(trajectory[-1])
            probability = transition_matrix[previous_location]
            # mask the locations that already in the trajectory
            for location in trajectory:
                probability[location] = 0
            probability = probability/sum(probability)
            next_location = np.random.choice(len(probability), size=1, p=probability)[0]
            trajectory.append(next_location)
            trajectory[i+1] = trajectory[reference[i+1]]
        trajectories.append(trajectory)
    return trajectories, times


def compute_distance(distance_matrix, pre_state, prob):
    return (distance_matrix[pre_state] * prob).sum()

def convert_distance_label_distribution(distance_labels, pre_state, probs, n_distance_labels):
    distance_label = distance_labels[pre_state]

    distance_label_distribution = torch.zeros(n_distance_labels).to(probs.device)
    # distance_label_distribution = distance_label_distribution.scatter_add(0, distance_label, probs)

    for label, prob in zip(distance_label, probs):
        distance_label_distribution[label] += prob
    return distance_label_distribution

# def compute_distance_distribution(pairs, distance_labels, n_distance_labels):
#     # compute distance
#     distances = []
#     for pre_state, prob in pairs:
#         distances.append(convert_distance_label_distribution(torch.tensor(distance_labels), pre_state, prob, n_distance_labels))

#     distances = torch.stack(distances)
#     # print(distances.sum())
#     # take mean
#     prob = torch.mean(distances, axis=0)
    
#     return prob

def compute_distance_distribution(pairs, distance_labels, n_distance_labels):
    # compute distance
    distance_label_distribution = torch.zeros(n_distance_labels).to(list(pairs.values())[0].device)
    for pre_state, prob in pairs.items():
        distance_label_distribution += convert_distance_label_distribution(distance_labels, pre_state, prob, n_distance_labels)

    # normalize
    distance_label_distribution = distance_label_distribution / distance_label_distribution.sum().item()
    
    return distance_label_distribution


def compute_distance_labels(distance_matrix, distance_to_label):
    labels = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]), dtype=int)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            labels[i, j] = distance_to_label(distance_matrix[i, j])
    return labels

def compute_distance_distribution_(pairs, distance_matrix, n_bins_for_distance):
    # compute distance
    distances = []
    for pre_state, prob in pairs:
        distances.append(compute_distance(distance_matrix, pre_state, prob))

    # binning distances
    bins = np.linspace(0, np.max(distances), n_bins_for_distance+1)
    binned_distances = np.digitize(distances, bins)

    # make probability distribution of binned_distances
    counter = Counter(binned_distances)
    prob = np.zeros(n_bins_for_distance)
    for i in range(n_bins_for_distance):
        prob[i] = counter[i+1] / len(distances)
    
    return prob

def load_time_dataset(data_dir, *, logger):


    if data_dir.parts[2] == "taxi":
        max_seq_len = 2
        real_time_traj = []
        data_path = data_dir / "training_data.csv"
        trajectories = load_dataset(data_path, logger=logger)
        for traj in trajectories:
            real_time_traj.append(list(range(len(traj))) + [max_seq_len])
    else:
        time_data_path = data_dir / "training_data_time.csv"
        real_time_traj = load_dataset(time_data_path, logger=logger)

    
    return real_time_traj


def load_dataset(data_dir, *, logger):
    trajectories = []

    # for data_dir in data_dirs:
    logger.info(f"load data from {data_dir}")
    data = pd.read_csv(data_dir, header=None).astype(str).values
    for trajectory in data:
        trajectory = [int(float(v)) for v in trajectory if (v != 'nan')]
        trajectories.append(trajectory)
    logger.info(f"length of dataset: {len(trajectories)}")

    if data_dir.parts[2] == "taxi":
        new = []
        for trajectory in trajectories:
            if len(trajectory) > 1 and trajectory[0] != trajectory[-1]:
                new.append([trajectory[0], trajectory[-1]])
            else:
                new.append([trajectory[0]])
        trajectories = new
    return trajectories
                 
def get_datadir():
    with open(f"config.json", "r") as f:
        config = json.load(f)
    return pathlib.Path(config["data_dir"])


def get_gps(dataset):
    df = pd.read_csv(get_datadir() / f"{dataset}/gps.csv", header=None)
    return df.values[:,1], df.values[:,0]

def make_gps(lat_range, lon_range, n_bins):
    
    x_axis = np.linspace(lon_range[0], lon_range[1], n_bins+2)
    y_axis = np.linspace(lat_range[0], lat_range[1], n_bins+2)
    
    def state_to_latlon(state):
        x_state = int(state % (n_bins+2))
        y_state = int(state / (n_bins+2))
        return y_axis[y_state], x_axis[x_state]
    
    return pd.DataFrame([state_to_latlon(i) for i in range((n_bins+2)**2)])


def load_M1(dataset):
    return normalize(np.load(get_datadir() / f'{dataset}/M1.npy'))

def load_M2(dataset):
    return normalize(np.load(get_datadir() / f'{dataset}/M2.npy'))

def construct_M1(training_data, max_locs):
    reg1 = np.zeros([max_locs,max_locs])
    for line in training_data:
        for j in range(len(line)-1):
            if (line[j] >= max_locs) or (line[j+1] >= max_locs):
#                 print("WARNING: outside location found")
                continue
            reg1[line[j],line[j+1]] +=1
    return reg1

def construct_M2(train_data, max_locs, gps):
    xs = gps[0]
    ys = gps[1]

    reg2 = []
    for (x,y) in zip(xs,ys):
        reg2.append(np.sqrt((ys - y)**2 + (xs - x)**2))
    reg2 = np.array(reg2)  
    return reg2



def load_latlon_range(name):
    with open(f"{name}.json", "r") as f:
        configs = json.load(f)
    lat_range = configs["lat_range"]
    lon_range = configs["lon_range"]
    return lat_range, lon_range
    
def latlon_to_state(lat, lon, lat_range, lon_range, n_bins):
    x_axis = np.linspace(lon_range[0], lon_range[1], n_bins+1)
    y_axis = np.linspace(lat_range[0], lat_range[1], n_bins+1)
    
    x_state = bisect.bisect_left(x_axis, lon)
    y_state = bisect.bisect_left(y_axis, lat)
    # if x_state == n_bins+2:
    #     x_state = n_bins
    # if y_state == n_bins+2:
    #     y_state = n_bins
    return y_state*(n_bins+2) + x_state

def make_hist_2d(counts, n_bins):
    hist2d = [[0 for i in range(n_bins+2)] for j in range(n_bins+2)]
    for state in range((n_bins+2)**2):
        x,y = state_to_xy(state, n_bins)
        hist2d[x][y] = counts[state]
    return np.array(hist2d)

def state_to_xy(state, n_bins):
    n_x = n_bins+2
    n_y = n_bins+2

    x = (state) % n_x
    y = int((state) / n_y)
    return x, y

def split_train_test(df, seed, split_ratio=0.5):
    random.seed(seed)
    n_records = len(df.index)
    choiced_indice = random.sample(range(n_records), int(n_records*split_ratio))
    removed_indice = [i for i in range(n_records) if i not in choiced_indice]
    training_df = df.loc[choiced_indice]
    test_df = df.loc[removed_indice]
    return training_df, test_df


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def set_logger(__name__, save_path):
    with open('./log_config.json', 'r') as f:
        log_conf = json.load(f)
    log_conf["handlers"]["fileHandler"]["filename"] = str(save_path)
    config.dictConfig(log_conf)
    logger = getLogger(__name__)
    logger.info('log is saved to {}'.format(save_path))

    return logger


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


def laplace_mechanism(x, epsilon):
    if epsilon == 0:
        return x
    return x + np.random.laplace(0, 1/epsilon, len(x))

def noise_normalize(values):
    # negative values are set to 0
    values = np.array(values)
    values[values < 0] = 0
    # normalize the values
    summation = np.sum(values)
    if summation == 0:
        return None
    return [v / summation for v in values]

def global_clipping(trajectories, global_clip):
    # bound the trajectory length to global_clip by randomly choosing global_clip locations from the trajectory
    clipped_trajectories = []
    for trajectory in trajectories:
        if len(trajectory) > global_clip:
            clipped_trajectories.append(random.sample(trajectory, global_clip))
        else:
            clipped_trajectories.append(trajectory)
    return clipped_trajectories


def plot_density(next_location_distribution, n_locations, save_path):
    if np.sqrt(n_locations).is_integer():
        n_x = int(np.sqrt(n_locations))
        n_y = int(np.sqrt(n_locations))
        values = np.array(next_location_distribution).reshape(n_x, n_y)
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(values, cmap="YlGnBu", vmin=0, vmax=values.max(), square=True, cbar_kws={"shrink": 0.8})
        plt.savefig(save_path)
        plt.close()
    else:
        # print("cannot plot density because n_locations is not a square number")
        pass

def add_noise(values, sensitivity, epsilon):
    # add Laplace noise
    if epsilon == 0:
        return values
    values = np.array(values) + np.random.laplace(loc=0, scale=sensitivity/epsilon, size=len(values))
    return values.tolist()


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


# clustering based on the global distribution and distance
# each cluster has the probability that is approximately larger than 1/n_classes
# basically, locations in the same cluster are closer to each other than locations in different clusters
def clustering(global_distribution, distance_matrix, n_classes):

    def check_threshold(clustered_in_i, global_distribution, threshold):
        # check if the sum of probability of locations in clustered_in_i is larger than threshold
        sum_prob = 0
        for loc in clustered_in_i:
            sum_prob += global_distribution[loc]
        return sum_prob > threshold

    location_to_class = {}
    threshold = 1/n_classes
    clustered = []
    all_locations = list(range(len(global_distribution)))
    # find locations to be clustered to class i
    for i in range(n_classes):
        # base_location is the smallest id of location that is not clustered yet
        remained = set(all_locations) - set(clustered)
        if len(remained) == 0:
            print('all locations are clustered')
            break
        else:
            base_location = min(set(all_locations) - set(clustered))
        clustered_in_i = [base_location]
        location_to_class[base_location] = i
        # sort locations by distance to base_location to find the closest location
        sorted_locations = sorted(set(all_locations) - set(clustered_in_i+clustered), key=lambda x: distance_matrix[x][base_location])
        for loc in sorted_locations:
            # check if the sum of probability of locations in clustered_in_i is larger than threshold
            if check_threshold(clustered_in_i, global_distribution, threshold):
                # print(f"prob for class {i} is {sum([global_distribution[loc] for loc in clustered_in_i])}")
                break
            # add the closest location to clustered_in_i
            clustered_in_i.append(loc)
            location_to_class[loc] = i
        clustered.extend(clustered_in_i)
    
    # assign the rest of locations to the last class
    for loc in set(all_locations) - set(clustered):
        location_to_class[loc] = n_classes-1
        
    return location_to_class