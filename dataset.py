from torch.utils.data import Dataset
import torch
import random
import numpy as np


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
    

def compute_traj_type_distribution(real_traj):
    # make dictionary that maps a format to a label
    format_to_label = make_format_to_label(real_traj)
    # label_to_format
    label_to_format = make_label_to_format(format_to_label)

    # make a list of labels
    label_list = [format_to_label[traj_to_format(trajectory)] for trajectory in real_traj]

    # count the number of trajectories for each label
    label_count = [0] * len(format_to_label)
    for label in label_list:
        label_count[label] += 1
    
    # normalize
    label_count = [count / sum(label_count) for count in label_count]
    return label_count, format_to_label, label_to_format

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

def padded_collate(batch):
    # compute max_len
    max_len = max([len(x["input"]) for x in batch])
    inputs = []
    targets = []
    for record in batch:
        s = record["input"]
        target = record["target"]
        inputs.append(s + [dataset.IGNORE_IDX] * (max_len - len(s)))
        targets.append(target + [dataset.IGNORE_IDX] * (max_len - len(target)))

    return {"input":torch.Tensor(inputs).long(), "target":torch.Tensor(targets).long()}

def padded_collate_without_end(batch):
    # compute max_len
    max_len = max([len(x["input"]) for x in batch])-1
    if max_len == 0:
        max_len = 1
    inputs = []
    targets = []
    for record in batch:
        # remove elements with length 1
        if len(record["input"]) == 1:
            continue

        s = record["input"][:-1]
        target = record["target"][:-1]
        inputs.append(s + [dataset.IGNORE_IDX] * (max_len - len(s)))
        targets.append(target + [dataset.IGNORE_IDX] * (max_len - len(target)))

    return {"input":torch.Tensor(inputs).long(), "target":torch.Tensor(targets).long()}


def padded_collate_without_end_only_first_markov(batch):
    # compute max_len
    inputs = []
    targets = []
    for record in batch:
        if len(record["input"]) < 2:
            continue

        s = record["input"][:-1][:1]
        target = record["target"][:-1][:1]
        inputs.append(s)
        targets.append(target)

    return {"input":torch.Tensor(inputs).long(), "target":torch.Tensor(targets).long()}


def make_padded_collate(n_locations, format_to_label, time_to_label, remove_first_value=False):
    start_idx = TrajectoryDataset.start_idx(n_locations)
    ignore_idx = TrajectoryDataset.ignore_idx(n_locations)

    def padded_collate(batch):
        # compute max_len
        max_len = max([len(x["trajectory"]) for x in batch])
        inputs = []
        targets = []
        times = []
        target_times = []
        labels = []

        for record in batch:
            trajectory = record["trajectory"]
            time_trajecotry = record["time_trajectory"]

            format = traj_to_format(trajectory)
            label = format_to_label[format]

            # print(trajectory)

            input = [start_idx] + trajectory + [ignore_idx] * (max_len - len(trajectory))
            target = input[1:] + [ignore_idx]

            # convert the duplicated state of target to the ignore_idx
            # if the label is "010", then the second 0 is converted to the ignore_idx
            checked_target = ["a"]
            for i in range(1,len(format)):
                if format[i] not in checked_target:
                    checked_target.append(format[i])
                    continue
                target[i] = ignore_idx

            if remove_first_value:
                target[0] = ignore_idx

            # convert time_input to labels [0,800,1439,...] -> [0, 5, 10, ...]
            time_last = time_to_label(-1)
            time_input = [time_to_label(v) for v in time_trajecotry] + [time_last] * (max_len - len(time_trajecotry)+1)
            time_target = time_input[1:] + [time_last]
            # if len(time_input) != len(input):
            #     print("aa", time_input)
            #     print("bb", input)

            inputs.append(input)
            targets.append(target)
            times.append(time_input)
            target_times.append(time_target)
            labels.append(label)

        # print([len(v) for v in inputs])
        # print([len(v) for v in targets])
        # print([len(v) for v in times])
        # print([len(v) for v in target_times])
        return {"input":torch.Tensor(inputs).long(), "target":torch.Tensor(targets).long(), "time":torch.Tensor(times).long(), "time_target":torch.Tensor(target_times).long(), "label":torch.Tensor(labels).long()}



        # for record in batch:
        #     if len(record["input"]) == 1:
        #         continue
        #     label = record["label"]
        #     s = record["input"][:-1]
        #     target = record["target"][:-1]
        #     time = convert_time_traj_to_time_traj_float(record["time"])[:-1]
        #     target_time = convert_time_traj_to_time_traj_float(record["time_target"][:-1])
        #     if len(s) != len(time):
        #         continue
        #     inputs.append([start_idx] + s + [ignore_idx] * (max_len - len(s)))
        #     targets.append(target + [ignore_idx] * (max_len - len(target)))
        #     times.append(time + [ignore_idx] * (max_len - len(time)))
        #     target_times.append(target_time + [ignore_idx] * (max_len - len(target_time)))
        #     labels.append(label)

    return padded_collate





# convert minute to float of [0,1]
def int_to_float_of_minute(minute):
    return minute / 1439

# convert real_time_traj to real_time_traj_float
def convert_time_traj_to_time_traj_float(real_time_traj):
    real_time_traj_float = []
    for time_start, _ in real_time_traj:
        real_time_traj_float.append(int_to_float_of_minute(time_start))
    return real_time_traj_float


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
    def time_to_label(time, n_time_split, max_time):
        if time == -1 or time == max_time:
            return n_time_split
        if time == 0:
            return 0
        return int(time//(max_time/n_time_split))+1
    
    def _time_to_label(self, time):
        return TrajectoryDataset.time_to_label(time, self.n_time_split, self.max_time)

    @staticmethod
    def label_to_time(label, n_time_split, max_time):
        return int(label*max_time/n_time_split)
    
    def _label_to_time(self, label):
        return TrajectoryDataset.label_to_time(label, self.n_time_split, self.max_time)
    
    #Init dataset
    def __init__(self, data, time_data, n_bins, n_time_split, max_time, dataset_name="dataset"):
        dataset = self
        
        dataset.data = data
        dataset.seq_len = max([len(trajectory) for trajectory in data])

        dataset.time_data = time_data

        dataset.n_bins = n_bins
        dataset.n_locations = (n_bins+2)**2
        # vocab = list(range(dataset.n_locations)) + ['<start>', '<ignore>', '<end>', '<oov>', '<mask>', '<cls>']
        # dataset.vocab = {e:i for i, e in enumerate(vocab)} 
        dataset.dataset_name = dataset_name
        dataset.time_end_index = n_time_split-1
        dataset.label_count, dataset.format_to_label, dataset.label_to_format = compute_traj_type_distribution(data)
        dataset.labels = self._compute_dataset_labels()
        dataset.n_time_split = n_time_split
        dataset.max_time = max_time
        
        #special tags
        # dataset.START_IDX = dataset.vocab['<start>']
        # dataset.IGNORE_IDX = dataset.vocab['<ignore>'] #replacement tag for tokens to ignore
        # dataset.OUT_OF_VOCAB_IDX = dataset.vocab['<oov>'] #replacement tag for unknown words
        # dataset.MASK_IDX = dataset.vocab['<mask>'] #replacement tag for the masked word prediction task
        # dataset.CLS_IDX = dataset.vocab['<cls>']
        # dataset.END_IDX = dataset.vocab['<end>']
    
    def __str__(self):
        return self.dataset_name
        
    # fetch data
    def __getitem__(self, index):
        dataset = self
        # s = dataset.data[index]
        # target = dataset.data[index][1:] + [dataset.END_IDX]
        trajectory = dataset.data[index]
        time_trajectory = dataset.time_data[index]
        

        return {'trajectory': trajectory, 'time_trajectory': time_trajectory}
        # return {'input': s, 'target': target, 'time': time, 'time_target': time_target, 'index': index, 'label': label}
        # return {'input':torch.Tensor(s).long(), 'target':torch.Tensor(target).long()}

    def __len__(self):
        return len(self.data)
    
    def _compute_dataset_labels(self):
        labels = [self.format_to_label[traj_to_format(trajectory)] for trajectory in self.data]
        return labels

    def reset(self):
        self.start_positions = {}

    #get words id
    def get_sentence_idx(self, index):
        dataset = self