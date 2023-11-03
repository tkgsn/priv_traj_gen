from torch.utils.data import Dataset
import torch
import random
import numpy as np
from my_utils import plot_density
from evaluation import compute_next_location_count, compute_global_counts_from_time_label, count_passing_locations, count_source_locations, count_target_locations, count_route_locations, count_distance
from collections import Counter
import json
import tqdm
import pathlib

def make_format_to_label(traj_list):
    format_to_label = {}
    for trajectory in traj_list:
        traj_type = traj_to_format(trajectory)
        if traj_type not in format_to_label:
            format_to_label[traj_type] = len(format_to_label)
    return format_to_label

def make_label_to_format(format_to_label):
    label_to_format = {}
    for format in format_to_label:
        label = format_to_label[format]
        label_to_format[label] = format
    return label_to_format
    

def make_label_info(real_traj):
    # make dictionary that maps a format to a label
    format_to_label = make_format_to_label(real_traj)
    # label_to_format
    label_to_format = make_label_to_format(format_to_label)

    return format_to_label, label_to_format

def traj_to_format(traj):
    # list a set of states in the trajectory
    # i.e., remove the duplicated states
    states = []
    for state in traj:
        if state not in states:
            states.append(state)
    # convert the list of states to a string
    # i.e., convert the list of states to a format
    format = ''
    for state in traj:
        # convert int to alphabet
        format += chr(states.index(state) + 97)
        # format += str(states.index(state))

    return format

class TrajectoryDataset(Dataset):

    @staticmethod
    def start_idx(n_locations):
        return n_locations
    
    @staticmethod
    def ignore_idx(n_locations):
        return n_locations+1
    
    @staticmethod
    def end_idx(n_locations):
        return n_locations+2
    
    @staticmethod
    def vocab_size(n_locations):
        return n_locations+3
    
    @staticmethod
    def time_end_idx(n_split):
        return n_split

    def _time_end_idx(self):
        return TrajectoryDataset.time_end_idx(self.n_time_split)
    
    @staticmethod
    def time_to_label(time, n_time_split, max_time):
        if time == 0:
            return 0
        elif time == max_time:
            return n_time_split
        assert time <= max_time, f"time {time} is larger than max_time {max_time}"
        return int(time//(max_time/n_time_split))+1
    
    def _time_to_label(self, time):
        return TrajectoryDataset.time_to_label(time, self.n_time_split, self.max_time)

    @staticmethod
    def label_to_time(label, n_time_split, max_time):
        return max([int((label-1)*max_time/n_time_split),0])
    
    def _label_to_time(self, label):
        return TrajectoryDataset.label_to_time(label, self.n_time_split, self.max_time)
    
    def label_to_length(self, label):
        format = self.label_to_format[label]
        return len(format)
    
    def make_reference(self, label):
        format = self.label_to_format[label]
        reference = {}
        for i in range(len(format)):
            if format[i] not in reference:
                reference[format[i]] = i
                
        return tuple([reference[format[i]] for i in range(len(format))])
    
    def _make_label_to_reference(self):
        
        return {label: self.make_reference(label) for label in self.label_to_format.keys()}
    
    #Init dataset
    def __init__(self, data, time_data, n_locations, n_time_split, real_start=True, dataset_name="dataset", route_data=None):
        assert len(data) == len(time_data)
        
        self.data = data
        self.seq_len = max([len(trajectory) for trajectory in data])
        self.time_data = time_data
        self.n_locations = n_locations
        self.dataset_name = dataset_name
        self.format_to_label, self.label_to_format = make_label_info(data)
        self.labels = self._compute_dataset_labels()
        self.label_to_reference = self._make_label_to_reference()
        self.reference_to_label_ = {reference: label for label, reference in self.label_to_reference.items()}
        self.references = [self.label_to_reference[label] for label in self.labels]
        if real_start:
            self.references = [tuple([traj[0]] + list(reference[1:])) for reference, traj in zip(self.references, self.data)]
        else:
            self.references = [tuple([-1] + list(reference[1:])) for reference in self.references]
        self.n_time_split = n_time_split
        self.max_time = max([max(time_traj) for time_traj in time_data])

        self.time_label_trajs = []
        for time_traj in self.time_data:
            self.time_label_trajs.append(tuple([self._time_to_label(t) for t in time_traj]))

        self.time_ranges = [(self._label_to_time(i), self._label_to_time(i+1)) for i in range(n_time_split)]
        self.computed_auxiliary_information = False

        self.n_bins_for_distance = 30

        if route_data is not None:
            assert len(route_data) == len(data)
            print("route data is given")
            self.route_data = route_data

        else:
            print("route_data is not given")
            self.route_data = data
            print("use trajectory data as route data")

    def reference_to_label(self, reference):
        reference = tuple([0] + list(reference[1:]))
        return self.reference_to_label_[reference]

    def __str__(self):
        return self.dataset_name
        
    # fetch data
    def __getitem__(self, index):
        trajectory = self.data[index]
        time_trajectory = list(self.time_label_trajs[index])
        
        return {'trajectory': trajectory, 'time_trajectory': time_trajectory}

    def __len__(self):
        return len(self.data)
    
    def _compute_dataset_labels(self):
        labels = [self.format_to_label[traj_to_format(trajectory)] for trajectory in self.data]
        return labels
    
    def convert_time_label_trajs_to_time_trajs(self, time_label_trajs):
        time_trajs = []
        for time_label_traj in time_label_trajs:
            time_trajs.append(self.convert_time_label_traj_to_time_traj(time_label_traj))
        return time_trajs
    
    def convert_time_label_traj_to_time_traj(self, time_label_traj):
        time_traj = []
        for time_label in time_label_traj:
            time_traj.append(self._label_to_time(time_label))
        return time_traj

    def make_next_location_count(self, target_index, order=1):
        if order == 1:
            print(f"compute {target_index} next location count")
            # compute the next location probability for each location
            next_location_counts = {}
            for label, traj in tqdm.tqdm(zip(self.labels, self.data)):
                reference = self.label_to_reference[label]

                if target_index == 0:
                    # count next location by the marginal way
                    for i in range(1,len(reference)):
                        if not (reference[i] == max(reference[:i+1])):
                            continue
                        if traj[i-1] not in next_location_counts:
                            next_location_counts[traj[i-1]] = [0 for _ in range(self.n_locations)]
                        next_location_counts[traj[i-1]][traj[i]] += 1
                elif target_index == 1:
                    # count next location by the first next location
                    if len(reference) < 2:
                        continue
                    if traj[0] not in next_location_counts:
                        next_location_counts[traj[0]] = [0 for _ in range(self.n_locations)]
                    next_location_counts[traj[0]][traj[1]] += 1
                elif target_index == 2:
                    # count next location by the second next location
                    if len(reference) < 3:
                        continue
                    if reference[2] != 2:
                        continue
                    if traj[1] not in next_location_counts:
                        next_location_counts[traj[1]] = [0 for _ in range(self.n_locations)]
                    next_location_counts[traj[1]][traj[2]] += 1

        elif order == 2:
            print(f"compute {target_index} first second order next location count")
            # compute the next location probability for each location
            next_location_counts = {}
            for label, traj in tqdm.tqdm(zip(self.labels, self.data)):
                reference = self.label_to_reference[label]
                if len(reference) < 3:
                    continue
                if reference[2] != 2:
                    continue

                if (traj[0], traj[1]) not in next_location_counts:
                    next_location_counts[(traj[0], traj[1])] = [0 for _ in range(self.n_locations)]
                next_location_counts[(traj[0], traj[1])][traj[2]] += 1

        return next_location_counts

    def compute_auxiliary_information(self, save_path, logger):

        (save_path.parent / "imgs").mkdir(exist_ok=True)
        if not self.computed_auxiliary_information:

            # compute top_base_locations
            self.first_locations = [trajectory[0] for trajectory in self.data if len(trajectory) > 1]
            self.first_location_counts = Counter(self.first_locations)
            self.route_first_locations = [trajectory[0] for trajectory in self.route_data if len(trajectory) > 1]
            self.route_first_location_counts = Counter(self.route_first_locations)
            self.top_base_locations = sorted(self.first_location_counts, key=lambda x: self.first_location_counts[x], reverse=True)
            self.top_route_base_locations = sorted(self.route_first_location_counts, key=lambda x: self.route_first_location_counts[x], reverse=True)
            # compute the next location counts
            self.next_location_counts = self.make_next_location_count(0)
            self.first_next_location_counts = self.make_next_location_count(1)
            self.second_next_location_counts = self.make_next_location_count(2)
            self.second_order_next_location_counts = self.make_next_location_count(0, order=2)

            # compute the global counts
            real_global_counts = []
            for time in range(1,self.n_time_split+1):
                real_global_count = Counter({location:0 for location in range(self.n_locations)})
                real_global_count.update(compute_global_counts_from_time_label(self.data, self.time_label_trajs, time))
                real_global_count = list(real_global_count.values())
                real_global_counts.append(real_global_count)
                if sum(real_global_count) == 0:
                    logger.info(f"no location at time {time}")
                    continue
                plot_density(real_global_count, self.n_locations, save_path.parent / "imgs" / f"real_global_distribution_{int(time)}.png")
            global_counts_path = save_path.parent / f"global_count.json"
            # save the global counts
            with open(global_counts_path, "w") as f:
                json.dump(real_global_counts, f)
            self.global_counts = real_global_counts

            # make a list of labels
            label_list = [self.format_to_label[traj_to_format(trajectory)] for trajectory in self.data]
            label_count = Counter({label:0 for label in self.label_to_format.keys()})
            label_count.update(label_list)
            reference_distribution = {self.label_to_reference[label]: count for label, count in label_count.items()}
            
            # compute time distribution
            time_label_count = Counter(self.time_label_trajs)
            time_distribution = {label: time_label_count[label] / len(self.time_label_trajs) for label in time_label_count.keys()}

            # compute counters
            self.real_counters = {}
            self.real_counters["global"] = [Counter({key:count for key, count in enumerate(global_count)}) for global_count in real_global_counts]
            self.real_counters["passing"] = count_passing_locations(self.route_data)
            self.real_counters["source"] = count_source_locations(self.data)
            self.real_counters["target"] = [count_target_locations(self.data, location) for location in self.top_base_locations]
            self.real_counters["route"] = [count_route_locations(self.route_data, location) for location in self.top_base_locations]
            self.real_counters["destination"] = [count_route_locations(self.data, location) for location in self.top_base_locations]
            logger.info(f"load distance matrix from" + str(pathlib.Path("/data") / str(save_path).split("/")[3]  / f"distance_matrix_bin{int(np.sqrt(self.n_locations)) -2}.npy"))
            self.distance_matrix = np.load(pathlib.Path("/data") / str(save_path).split("/")[3]  / f"distance_matrix_bin{int(np.sqrt(self.n_locations)) -2}.npy")
            self.real_counters["distance"] = count_distance(self.distance_matrix, self.data, self.n_bins_for_distance)

            # compute n_trajs
            self.n_trajs = {}
            self.n_trajs["global"] = [len(self.data) for global_count in real_global_counts]
            self.n_trajs["passing"] = len(self.data)
            self.n_trajs["source"] = len(self.data)
            self.n_trajs["target"] = [self.first_location_counts[location] for location in self.top_base_locations]
            self.n_trajs["route"] = [self.route_first_location_counts[location] for location in self.top_base_locations]
            self.n_trajs["destination"] = [self.first_location_counts[location] for location in self.top_base_locations]
            self.n_trajs["distance"] = len(self.data)

            self.computed_auxiliary_information = True
        # return locations, next_location_counts, first_next_location_counts, real_global_counts, label_count, time_distribution, reference_distribution



    def make_second_order_test_data_loader(self, n_test_locations):

        assert self.computed_auxiliary_information, "auxiliary information has not been computed"

        second_order_next_location_counts = self.second_order_next_location_counts
        n_test_locations = min(n_test_locations, len(second_order_next_location_counts))
        top_second_order_base_locations = sorted(second_order_next_location_counts, key=lambda x: sum(second_order_next_location_counts[x]), reverse=True)[:n_test_locations]

        # retrieving the trajectories that start with the first_location_counts
        counters = {}
        trajs = []
        time_trajs = []
        first_second_locations = top_second_order_base_locations[:n_test_locations]
        for first_location in first_second_locations:
            trajs_for_the_first_location = []
            time_trajs_for_the_first_location = []
            for traj, time_traj in zip(self.data, self.time_data):
                if len(traj) > 2:
                    if traj[0] == first_location[0] and traj[1] == first_location[1]:
                        trajs_for_the_first_location.append(traj)
                        time_trajs_for_the_first_location.append(time_traj)
            counters[first_location] = len(trajs_for_the_first_location)
            trajs.extend(trajs_for_the_first_location)
            time_trajs.extend(time_trajs_for_the_first_location)
        
        print(f"number of test trajectories: {len(trajs)}")
        print(f"number of test trajectories that start with: {counters}")

        if len(trajs) == 0:
            print("no trajectory (>2) is found")
            self.second_order_test_data_loader = None
            self.second_counters = None
            return None, None
        else:

            second_order_test_dataset = TrajectoryDataset(trajs, time_trajs, self.n_locations, self.n_time_split)
            second_order_test_data_loader = torch.utils.data.DataLoader(second_order_test_dataset, num_workers=0, shuffle=False, pin_memory=True, batch_size=100, collate_fn=second_order_test_dataset.make_padded_collate())
            
            self.second_order_test_data_loader = second_order_test_data_loader
            self.second_counters = counters
            return second_order_test_data_loader, counters

    def make_first_order_test_data_loader(self, n_test_locations):

        assert self.computed_auxiliary_information, "auxiliary information has not been computed"

        top_base_locations = self.top_base_locations[:n_test_locations]
        n_test_locations = min(n_test_locations, len(top_base_locations))

        # retrieving the trajectories that start with the first_location_counts
        counters = {}
        trajs = []
        time_trajs = []
        first_locations = top_base_locations[:n_test_locations]
        for first_location in first_locations:
            trajs_for_the_first_location = []
            time_trajs_for_the_first_location = []
            for traj, time_traj in zip(self.data, self.time_data):
                if traj[0] == first_location and len(traj) > 1:
                    trajs_for_the_first_location.append(traj)
                    time_trajs_for_the_first_location.append(time_traj)
            counters[first_location] = len(trajs_for_the_first_location)
            trajs.extend(trajs_for_the_first_location)
            time_trajs.extend(time_trajs_for_the_first_location)
        
        print(f"number of test trajectories: {len(trajs)}")
        print(f"number of test trajectories that start with: {counters}")

        test_dataset = TrajectoryDataset(trajs, time_trajs, self.n_locations, self.n_time_split)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, shuffle=False, pin_memory=True, batch_size=100, collate_fn=test_dataset.make_padded_collate())

        self.first_order_test_data_loader = test_data_loader
        self.first_counters = counters
        return test_data_loader, counters


    def make_padded_collate(self, remove_first_value=False, remove_duplicate=False):
        start_idx = TrajectoryDataset.start_idx(self.n_locations)
        ignore_idx = TrajectoryDataset.ignore_idx(self.n_locations)
        time_end_idx = TrajectoryDataset.time_end_idx(self.n_time_split)

        def padded_collate(batch):
            max_len = max([len(x["trajectory"]) for x in batch])
            inputs = []
            targets = []
            times = []
            target_times = []
            references = []

            for record in batch:
                trajectory = record["trajectory"]
                time_trajecotry = record["time_trajectory"]

                format = traj_to_format(trajectory)
                label = self.format_to_label[format]
                reference = self.label_to_reference[label]

                input = [start_idx] + trajectory + [ignore_idx] * (max_len - len(trajectory))
                target = input[1:] + [ignore_idx]

                # convert the duplicated state of target to the ignore_idx
                # if the label is "010", then the second 0 is converted to the ignore_idx
                if remove_duplicate:
                    checked_target = ["a"]
                    for i in range(1,len(format)):
                        if format[i] not in checked_target:
                            checked_target.append(format[i])
                            continue
                        target[i] = ignore_idx

                if remove_first_value:
                    target[0] = ignore_idx

                time_input = time_trajecotry + [time_end_idx] * (max_len - len(time_trajecotry)+1)
                time_target = time_input[1:] + [time_end_idx]

                inputs.append(input)
                targets.append(target)
                times.append(time_input)
                target_times.append(time_target)
                references.append(reference)
            
            return {"input":torch.Tensor(inputs).long(), "target":torch.Tensor(targets).long(), "time":torch.Tensor(times).long(), "time_target":torch.Tensor(target_times).long(), "reference":references}

        return padded_collate
        