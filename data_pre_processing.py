from my_utils import make_gps
import json
import pathlib
import argparse
import pandas as pd
import numpy as np
from my_utils import get_datadir
from grid import Grid
import tqdm
from bisect import bisect_left
from geopy.distance import geodesic

def compute_distance_matrix(state_to_latlon, n_locations):
    distance_matrix = np.zeros((n_locations, n_locations))
    for i in tqdm.tqdm(range(n_locations)):
        for j in range(n_locations):
            distance_matrix[i, j] = geodesic(state_to_latlon(i), state_to_latlon(j)).meters
    return distance_matrix


def make_stay_trajectory(trajectories, time_threshold, location_threshold):

    print(f"make stay trajectory with threshold {location_threshold}m and {time_threshold}min")

    stay_trajectories = []
    time_trajectories = []
    for trajectory in tqdm.tqdm(trajectories):

        stay_trajectory = []
        # remove nan
        trajectory = [v for v in trajectory if type(v) is str]
        time_trajectory = []

        start_index = 0
        start_time = 0
        i = 0

        while True:
            # find the length of the stay
            start_location = trajectory[start_index].split(" ")
            start_location = (float(start_location[1]), float(start_location[2]))

            if i == len(trajectory)-1:
                time_trajectory.append((start_time, time))
                stay_trajectory.append(target_location)
                # print("finish", start_time, time, start_location)
                break

            for i in range(start_index+1, len(trajectory)):

                target_location = trajectory[i].split(" ")
                time = float(target_location[0])
                target_location = (float(target_location[1]), float(target_location[2]))
                distance = geodesic(start_location, target_location).meters
                if distance > location_threshold:
                    # print(f"move {distance}m", start_time, time, trajectory[i])
                    if time - start_time >= time_threshold:
                        # print("stay", start_time, time, start_location)
                        stay_trajectory.append(start_location)
                        time_trajectory.append((start_time, time))

                    start_time = time
                    # print(trajectory[i])
                    start_index = i
                    # print("start", start_time, start_index, len(trajectory))
                    # print(time, i)

                    break
        
        stay_trajectories.append(stay_trajectory)
        time_trajectories.append(time_trajectory)
    return time_trajectories, stay_trajectories


def make_complessed_dataset(time_trajectories, trajectories, grid):
    dataset = []
    times = []
    for trajectory, time_trajectory in tqdm.tqdm(zip(trajectories, time_trajectories)):
        state_trajectory = []
        for lat, lon in trajectory:
            state = grid.latlon_to_state(lat, lon)
            state_trajectory.append(state)

        if None in state_trajectory:
            continue

        # compless time trajectory according to state trajectory
        complessed_time_trajectory = []
        j = 0
        for i, time in enumerate(time_trajectory):
            if i != j:
                continue   
            target_state = state_trajectory[i]
            # find the max length of the same states
            for j in range(i+1, len(state_trajectory)+1):
                if j == len(state_trajectory):
                    break
                if (state_trajectory[j] != target_state):
                    break
            complessed_time_trajectory.append((time[0],time_trajectory[j-1][1]))

        # remove consecutive same states
        state_trajectory = [state_trajectory[0]] + [state_trajectory[i] for i in range(1, len(state_trajectory)) if state_trajectory[i] != state_trajectory[i-1]]

        dataset.append(state_trajectory)
        times.append(complessed_time_trajectory)

        assert len(state_trajectory) == len(complessed_time_trajectory), f"state trajectory length {len(state_trajectory)} != time trajectory length {len(complessed_time_trajectory)}"
        # times.append([time for time, _, _ in trajectory])
    return dataset, times

def save_with_nan_padding(save_path, trajectories, formater, verbose=False):
    # compute the max length in trajectories
    max_len = max([len(trajectory) for trajectory in trajectories])

    if verbose:
        print(f"save to {save_path}")
    with open(save_path, "w") as f:
        for trajectory in trajectories:
            for record in trajectory:
                f.write(formater(record))
            # padding with nan
            for _ in range(max_len - len(trajectory)):
                f.write(",")
            f.write("\n")

def save_timelatlon_with_nan_padding(save_path, trajectories):
    def formater(record):
        return f"{record[0]} {record[1]} {record[2]},"
    
    save_with_nan_padding(save_path, trajectories, formater)

def save_latlon_with_nan_padding(save_path, trajectories):
    def formater(record):
        return f"{record[1]} {record[2]},"
    
    save_with_nan_padding(save_path, trajectories, formater)

def save_state_with_nan_padding(save_path, trajectories, verbose=False):
    def formater(record):
        return f"{record},"
    
    save_with_nan_padding(save_path, trajectories, formater, verbose=verbose)


def save_time_with_nan_padding(save_path, trajectories, max_time):
    def formater(record):
        return f"{int(record[0])},"
    
    for trajectory in trajectories:
        trajectory.append([max_time])
    
    save_with_nan_padding(save_path, trajectories, formater)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--latlon_config', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--n_bins', type=int)
    parser.add_argument('--time_threshold', type=int)
    parser.add_argument('--location_threshold', type=int)
    parser.add_argument('--save_name', type=str)
    args = parser.parse_args()
    
    with open(pathlib.Path("./") / "dataset_configs" / args.latlon_config, "r") as f:
        configs = json.load(f)
    
    configs.update(vars(args))
    data_path = get_datadir() / args.dataset / args.data_name / args.save_name
    data_path.mkdir(exist_ok=True, parents=True)
        
    lat_range = configs["lat_range"]
    lon_range = configs["lon_range"]
    n_bins = args.n_bins
    time_threshold = args.time_threshold
    location_threshold = args.location_threshold

    if args.dataset == "chengdu":
        max_locs = 8263
    else:
        max_locs = (n_bins+2)**2
    max_time = 24*60-1

    # make training data
    training_data_paths = sorted(data_path.glob("training_data*.csv"))
    if training_data_paths == []:

        print("make grid", lat_range, lon_range, n_bins)
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        grid = Grid(ranges)

        raw_data_dir = data_path.parent
        # find csv file whose format is raw_data{i}.csv
        raw_data_paths = sorted(raw_data_dir.glob("raw_data*.csv"))
        print(raw_data_paths)
        # if configs["dataset"] == "geolife" or configs["dataset"] == "geolife_test":

        for i, raw_data_path in enumerate(raw_data_paths):
            save_path = data_path / f"training_data{i}.csv"
            print(f"save to {save_path}")
            print(f"load raw data from {raw_data_path}")
            trajs = pd.read_csv(raw_data_path, header=None).values
            
            print("make stay trajectory")
            time_trajs, trajs = make_stay_trajectory(trajs, time_threshold, location_threshold)
            print("make complessed dataset")
            dataset, times = make_complessed_dataset(time_trajs, trajs, grid)
            print(f"save complessed dataset to {save_path}")
            save_state_with_nan_padding(save_path, dataset)
            
            time_save_path = data_path / f"training_data_time{i}.csv"
            save_time_with_nan_padding(time_save_path, times, max_time)

    
    # make gps data
    if not (data_path/"gps.csv").exists():        
        gps = make_gps(lat_range, lon_range, n_bins)
        gps.to_csv(data_path / f"gps.csv", header=None, index=None)
    gps = pd.read_csv(data_path / f"gps.csv", header=None).values

    # make distance matrix
    if not (data_path/"distance_matrix.npy").exists():
        def state_to_latlon(state):
            return gps[state]
        distance_matrix = compute_distance_matrix(state_to_latlon, max_locs)
        np.save(data_path/f'distance_matrix.npy',distance_matrix)

    configs["n_locations"] = max_locs
    configs["max_time"] = max_time

    print("saving setting to", data_path / "params.json")
    with open(data_path / "params.json", "w") as f:
        json.dump(configs, f)