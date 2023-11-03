from my_utils import make_gps
import json
import pathlib
import argparse
import pandas as pd
import numpy as np
from my_utils import get_datadir, load, save
from grid import Grid
import tqdm
from bisect import bisect_left
from geopy.distance import geodesic
import make_pair_to_route

def compute_distance_matrix(state_to_latlon, n_locations):
    # compute the distance matrix using geodestic distance
    distance_matrix = np.zeros((n_locations, n_locations))
    for i in tqdm.tqdm(range(n_locations)):
        for j in range(i, n_locations):
            distance_matrix[i, j] = geodesic(state_to_latlon(i), state_to_latlon(j)).meters
            distance_matrix[j, i] = distance_matrix[i, j]

    # for i in tqdm.tqdm(range(n_locations)):
    #     for j in range(n_locations):
    #         distance_matrix[i, j] = geodesic(state_to_latlon(i), state_to_latlon(j)).meters
    return distance_matrix


def make_stay_trajectory(trajectories, time_threshold, location_threshold, startend=False):

    print(f"make stay-point trajectory with threshold {location_threshold}m and {time_threshold}min")

    stay_trajectories = []
    time_trajectories = []
    for trajectory in tqdm.tqdm(trajectories):

        if startend:
            trajectory = [trajectory[0], trajectory[-1]]
        stay_trajectory = [(trajectory[0][1], trajectory[0][2])]
        time_trajectory = [(trajectory[0][0], trajectory[0][0])]

        start_index = 0
        i = 0

        while True:
            # find the length of the stay
            start_record = trajectory[start_index]
            start_location = (start_record[1], start_record[2])
            start_time = start_record[0]

            if i == len(trajectory)-1:
                time_trajectory.append((start_time, time))
                stay_trajectory.append(target_location)
                # print("finish", start_time, time, start_location)
                break

            for i in range(start_index+1, len(trajectory)):

                target_location = trajectory[i]
                time = float(target_location[0])
                target_location = (target_location[1], target_location[2])
                distance = geodesic(start_location, target_location).meters
                if distance > location_threshold:
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
        # print(stay_trajectory, time_trajectory)

        
        stay_trajectories.append(stay_trajectory)
        time_trajectories.append(time_trajectory)
    
    if startend:
        time_trajectories = [[(i, i+1) for i in range(len(v))] for v in stay_trajectories]
    return time_trajectories, stay_trajectories


def make_complessed_dataset(time_trajectories, trajectories, grid):
    dataset = []
    times = []
    for ind in tqdm.tqdm(range(len(trajectories))):
        trajectory = trajectories[ind]
        time_trajectory = time_trajectories[ind]
        state_trajectory = []
        for lat, lon in trajectory:
            state = grid.latlon_to_state(lat, lon)
            state_trajectory.append(state)


        if None in state_trajectory:
            print("WARNING: FOUND OUT OF RANGE LOCATION, THE TRAJECTORY IS REMOVED")
            print(ind, trajectory)
            print(state_trajectory)
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

def check_in_range(trajs, grid):
    new_trajs = []
    for traj in tqdm.tqdm(trajs):
        new_traj = []
        for time, lat, lon in traj:
            new_traj.append(grid.is_in_range(lat, lon))
        if all(new_traj):
            new_trajs.append(traj)
    print(f"remove {len(trajs)-len(new_trajs)} trajectories")
    return new_trajs


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--latlon_config', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--n_bins', type=int)
    parser.add_argument('--time_threshold', type=int)
    parser.add_argument('--location_threshold', type=int)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--startend', action='store_true')
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


    max_locs = (n_bins+2)**2
    max_time = 24*60-1

    # make training data
    # training_data_paths = sorted(data_path.glob("training_data*.csv"))
    training_data_path = data_path / "training_data.csv"
    if not training_data_path.exists():

        print("make grid", lat_range, lon_range, n_bins)
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        grid = Grid(ranges)

        raw_data_dir = data_path.parent
        # find csv file whose format is raw_data{i}.csv
        # raw_data_paths = sorted(raw_data_dir.glob("raw_data*.csv"))
        raw_data_path = raw_data_dir / f"raw_data_seed{args.seed}.csv"
        # if configs["dataset"] == "geolife" or configs["dataset"] == "geolife_test":
        
        # raw_trajs = []
        # for i, raw_data_path in enumerate(raw_data_paths):
        print(f"load raw data from {raw_data_path}")
        raw_trajs = load(raw_data_path)
            # raw_trajs += trajs

        raw_trajs = check_in_range(raw_trajs, grid)
            
        print("make stay trajectory")
        time_trajs, trajs = make_stay_trajectory(raw_trajs, time_threshold, location_threshold, startend=args.startend)
        print("make complessed dataset")
        dataset, times = make_complessed_dataset(time_trajs, trajs, grid)

        save_path = data_path / f"training_data.csv"
        print(f"save complessed dataset to {save_path}")
        save(save_path, dataset)
        
        times = [[time[0] for time in traj] for traj in times]
        time_save_path = data_path / f"training_data_time.csv"
        print(f"save time dataset to {time_save_path}")
        save(time_save_path, times)

    
    # make gps data
    if not (data_path/"gps.csv").exists():        
        print("make gps data")
        gps = make_gps(lat_range, lon_range, n_bins)
        gps.to_csv(data_path / f"gps.csv", header=None, index=None)
    print("load gps data")
    gps = pd.read_csv(data_path / f"gps.csv", header=None).values

    # make distance matrix
    if not (data_path.parent.parent/f"distance_matrix_bin{n_bins}.npy").exists():
        print("make distance matrix")
        def state_to_latlon(state):
            return gps[state]
        distance_matrix = compute_distance_matrix(state_to_latlon, max_locs)
        print("save distance matrix to", data_path.parent.parent/f"distance_matrix_bin{n_bins}.npy")
        np.save(data_path.parent.parent/f"distance_matrix_bin{n_bins}.npy",distance_matrix)
    else:
        print("distance_matrix already exists")

    db_save_dir = data_path.parent.parent / "pair_to_route" / f"{n_bins}"
    db_save_dir.mkdir(exist_ok=True, parents=True)
    db_save_dir.mkdir(exist_ok=True, parents=True)
    if not (db_save_dir / "paths.db").exists():
    # if True:
        graph_data_dir = get_datadir() / args.dataset / args.data_name / "MTNet"
        latlon_config_path = f"./dataset_configs/{args.latlon_config}"

        print("make pair_to_route to",  db_save_dir)
        make_pair_to_route.run(n_bins, graph_data_dir, latlon_config_path, db_save_dir)
    else:
        print("pair_to_route already exists in", db_save_dir / "paths.db")
    

    configs["n_locations"] = max_locs
    configs["max_time"] = max_time

    print("saving setting to", data_path / "params.json")
    with open(data_path / "params.json", "w") as f:
        json.dump(configs, f)