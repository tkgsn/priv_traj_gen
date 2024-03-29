from my_utils import make_gps
import json
import pathlib
import argparse
import pandas as pd
import numpy as np
from my_utils import get_datadir, load, save, set_logger, load_latlon_range, get_original_dataset_name
from name_config import make_save_name
from grid import Grid
import tqdm
from bisect import bisect_left
from geopy.distance import geodesic
import make_pair_to_route
import concurrent.futures
import functools
from make_raw_data import make_raw_data_random, make_raw_data_rotation
import subprocess
import sqlite3
from name_config import make_training_data_path

def compute_distance_matrix(state_to_latlon, n_locations):

    partial_compute_distance = functools.partial(compute_distance_from_i, state_to_latlon=state_to_latlon, n_locations=n_locations)

    distance_matrix = np.zeros((n_locations, n_locations))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(partial_compute_distance, i) for i in range(n_locations)]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                state, distance_array = future.result()
                distance_matrix[state, :] = distance_array
            except Exception as exc:
                print(f'Generated an exception: {exc}')

    return distance_matrix

def compute_distance_from_i(state, state_to_latlon, n_locations):
    distance_array = np.zeros(n_locations)
    for i in range(n_locations):
        distance_array[i] = geodesic(state_to_latlon[state], state_to_latlon[i]).meters
    return state, distance_array

def process_trajectory(trajectory, location_threshold, time_threshold, startend):
    if startend:
        trajectory = [trajectory[0], trajectory[-1]]
    stay_trajectory = [(trajectory[0][1], trajectory[0][2])]
    time_trajectory = [(trajectory[0][0], trajectory[0][0])]

    if len(trajectory) == 1:
        return stay_trajectory, time_trajectory

    start_index = 0
    i = 0
    skip = (location_threshold == 0) and (time_threshold == 0)
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
            if skip or distance > location_threshold:
                if skip or time - start_time >= time_threshold*60:
                    # print("stay", start_time, time, start_location)
                    stay_trajectory.append(start_location)
                    time_trajectory.append((start_time, time))

                start_time = time
                # print(trajectory[i])
                start_index = i
                # print("start", start_time, start_index, len(trajectory))
                # print(time, i)

                break
    return stay_trajectory, time_trajectory


def make_stay_trajectory(trajectories, time_threshold, location_threshold, startend=False):

    print(f"make stay-point trajectory with threshold {location_threshold}m and {time_threshold}hour")

    partial_process_trajectory = functools.partial(process_trajectory, location_threshold=location_threshold, time_threshold=time_threshold, startend=startend)

    stay_trajectories = []
    time_trajectories = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(partial_process_trajectory, trajectories), total=len(trajectories)))
        
    stay_trajectories, time_trajectories = zip(*results)

    
    if startend:
        time_trajectories = [[(i, i+1) for i in range(len(v))] for v in stay_trajectories]
    return time_trajectories, stay_trajectories


def compless(state_trajectory, time_trajectory, cost=False):

    # def compless_2(traj):

    #     if len(traj) <= 3:
    #         return traj

    #     cursor = 2
    #     while True:
    #         if cursor > len(traj)-2:
    #             break

    #         if [traj[cursor-2],traj[cursor-1]] == [traj[cursor],traj[cursor+1]]:
    #             traj.pop(cursor)
    #             traj.pop(cursor)
    #             # cursor -= 1
    #         else:
    #             cursor += 1
        
    #     return traj

    # compless time trajectory according to state trajectory
    complessed_time_trajectory = []
    j = 0
    for i, time in enumerate(time_trajectory[:len(state_trajectory)]):
        if i != j:
            continue   
        target_state = state_trajectory[i]
        # find the max length of the same states
        for j in range(i+1, len(state_trajectory)+1):
            if j == len(state_trajectory):
                break
            if (state_trajectory[j] != target_state):
                break
        if cost:
            complessed_time_trajectory.append(sum(time_trajectory[i:j]))
        else:
            complessed_time_trajectory.append((time[0],time_trajectory[j-1][1]))
    # print("before", state_trajectory)
    # remove consecutive same states
    state_trajectory = [state_trajectory[0]] + [state_trajectory[i] for i in range(1, len(state_trajectory)) if state_trajectory[i] != state_trajectory[i-1]]

    # state_trajectory = compless_2(state_trajectory)
    # complessed_time_trajectory = [0] * len(state_trajectory)
    # print("after", state_trajectory)
    return state_trajectory, complessed_time_trajectory



def make_complessed_dataset(time_trajectories, trajectories, grid, indice=None):
    dataset = []
    times = []
    if indice is None:
        indice = range(len(trajectories))
    added_indice = []

    for ind in tqdm.tqdm(range(len(trajectories))):
        if ind not in indice:
            continue

        trajectory = trajectories[ind]
        time_trajectory = time_trajectories[ind]
        state_trajectory = []
        for lat, lon in trajectory:
            state = grid.latlon_to_state(lat, lon)
            state_trajectory.append(state)


        # if None in state_trajectory:
        #     print("WARNING: FOUND OUT OF RANGE LOCATION, THE TRAJECTORY IS REMOVED")
        #     print(ind, trajectory)
        #     print(state_trajectory)
        #     continue

        state_trajectory, complessed_time_trajectory = compless(state_trajectory, time_trajectory)

        if len(state_trajectory) <= 1:
            # print("removed because of length")
            # print(state_trajectory)
            continue

        dataset.append(state_trajectory)
        times.append(complessed_time_trajectory)
        added_indice.append(ind)

        assert len(state_trajectory) == len(complessed_time_trajectory), f"state trajectory length {len(state_trajectory)} != time trajectory length {len(complessed_time_trajectory)}"
        # times.append([time for time, _, _ in trajectory])
    return dataset, times, added_indice

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

def make_gps_data(training_data_dir, lat_range, lon_range, n_bins):
    if not (training_data_dir / "gps.csv").exists():        
        gps = make_gps(lat_range, lon_range, n_bins)
        gps.to_csv(training_data_dir / f"gps.csv", header=None, index=None)
    gps = pd.read_csv(training_data_dir / f"gps.csv", header=None).values
    # send(training_data_dir / f"gps.csv")
    return gps

def make_distance_data(training_data_dir, n_bins, gps, logger):
    # make distance matrix
    if not (training_data_dir.parent.parent / f"distance_matrix_bin{n_bins}.npy").exists():
        logger.info("make distance matrix")
        state_to_latlon = gps
        distance_matrix = compute_distance_matrix(state_to_latlon, (n_bins+2)**2)
        name = f'distance_matrix_bin{n_bins}.npy'
        logger.info(f"save distance matrix to {training_data_dir.parent.parent / name}")
        np.save(training_data_dir.parent.parent/f"distance_matrix_bin{n_bins}.npy",distance_matrix)
    else:
        logger.info("distance_matrix already exists")
    # send(training_data_dir.parent.parent / f"distance_matrix_bin{n_bins}.npy")

def make_db(dataset, lat_range, lon_range, n_bins, truncate, logger):

    original_dataset = get_original_dataset_name(dataset)

    db_save_dir = get_datadir() / original_dataset / "pair_to_route" / f"{n_bins}_tr{truncate}"
    db_save_dir.mkdir(exist_ok=True, parents=True)
    if not (db_save_dir / "paths.db").exists():
    # if True:
        # to avoid the latency of the shared file directory, we use temp_db_save_dir, which is moved to db_save_dir after
        temp_db_save_dir = "./temp"
        graph_data_dir = get_datadir() / dataset / "raw"
        logger.info(f"make pair_to_route to {db_save_dir}")
        make_pair_to_route.run(n_bins, graph_data_dir, lat_range, lon_range, truncate, temp_db_save_dir)
        # move to db_save_dir with subprocess
        logger.info(f"move {temp_db_save_dir} to {db_save_dir}")
        # (pathlib.Path(temp_db_save_dir) / "paths.db").rename(db_save_dir / "paths.db")
        subprocess.run(["mv", pathlib.Path(temp_db_save_dir) / "paths.db", db_save_dir / "paths.db"])
        # remove ./temp
        logger.info(f"remove {temp_db_save_dir}")
        subprocess.run(["rm", "-rf", temp_db_save_dir])
    else:
        logger.info(f"pair_to_route already exists in {db_save_dir / 'paths.db'}")
    
    return db_save_dir / "paths.db"
    
    # send(db_save_dir / "paths.db")

def make_reversible_stay_traj(traj, road_db):
    # open database
    with sqlite3.connect(road_db) as conn:
        reversible_stay_traj = [traj[0]]
        c = conn.cursor()
        cursor1 = 0
        while True:
            for cursor2 in range(cursor1+1, len(traj)+1):
                if cursor2 == len(traj):
                    reversible_stay_traj.append(traj[cursor2-1])
                    cursor1 = cursor2
                    break

                from_state = traj[cursor1]
                to_state = traj[cursor2]
                query = f"SELECT route FROM state_edge_to_route WHERE start_state={from_state} AND end_state={to_state}"
                c.execute(query)
                route = c.fetchone()
                # if route is the same as the partial traj from cursor to cursor2, then we can reverse the traj
                if route is not None:
                    route = eval(route[0])
                    # print(route, traj[cursor1:cursor2+1])
                    if route == traj[cursor1:cursor2+1]:
                        continue
                
                assert cursor2-1 != cursor1, f"the adjacent states should be connected by the road network, but {traj[cursor1]} and {traj[cursor2]} are not connected {route}"
                reversible_stay_traj.append(traj[cursor2-1])
                cursor1 = cursor2-1
                break
            if cursor1 == len(traj):
                break
    return reversible_stay_traj


def make_reversible_trajs(trajs, road_db):
    reversible_trajs = []
    for traj in tqdm.tqdm(trajs):
        reversible_trajs.append(make_reversible_stay_traj(traj, road_db))
    return reversible_trajs

def run(dataset_name, lat_range, lon_range, n_bins, time_threshold, location_threshold, size, seed, truncate, logger):
    """
    training_data is POI_id (aka state) trajectory
    state is made by grid of n_bins, which means there are (n_bins+2)*(n_bins+2) states in lat_range and lon_range
    and state trajectory is converted to stay-point trajectory by time_threshold and location_threshold
    then, the sequential duplication is removed
    """

    save_name = make_save_name(dataset_name, n_bins, time_threshold, location_threshold, seed)
    # training_data_dir = get_datadir() / dataset_name / f"{size}" / save_name
    training_data_dir = make_training_data_path(dataset_name, size, save_name)
    training_data_dir.mkdir(exist_ok=True, parents=True)

    if not (training_data_dir / "training_data.csv").exists():

        if dataset_name == "rotation":
            trajs, times = make_raw_data_rotation(seed, size, n_bins)
        
        elif dataset_name == "random":
            trajs, times = make_raw_data_random(seed, size, n_bins)

        else:
            # training_data_dir = get_datadir() / dataset_name / f"{size}" / f"{n_bins}_tr{truncate}"

            # training_data_dir.mkdir(exist_ok=True, parents=True)

            raw_data_path = training_data_dir.parent.parent / f"raw_data.csv"
            logger.info(f"load raw data from {raw_data_path}")
            raw_trajs = load(raw_data_path, size, seed)

            logger.info(f"make grid lat {lat_range} lon {lon_range} n_bins {n_bins}")
            ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
            grid = Grid(ranges)
            logger.info(f"check in range")
            raw_trajs = check_in_range(raw_trajs, grid)

            if dataset_name in ["chengdu", "geolife_mm"]:
                # for the road network dataset, we make stay trajectory with road network information
                db_path = make_db(dataset_name, lat_range, lon_range, n_bins, truncate, logger)

                # represent route trajectory by states
                route_time_trajs, route_trajs = make_stay_trajectory(raw_trajs, 0, 0)
                route_trajs, route_time_trajs, indice = make_complessed_dataset(route_time_trajs, route_trajs, grid)
                # copy db_path to ./temp
                subprocess.run(["cp", db_path, "./"])
                # convert to reversible stay trajectory from the route trajectory
                print("make reversible trajs using", db_path)
                print(route_trajs[:10])
                trajs = make_reversible_trajs(route_trajs, "./paths.db")
                times = [[0 for _ in traj] for traj in trajs]

                logger.info(f"save route complessed dataset to {training_data_dir / f'route_training_data.csv'}")
                save(training_data_dir / f"route_training_data.csv", route_trajs)
                logger.info(f"save route time dataset to {training_data_dir / f'route_training_data_time.csv'}")
                save(training_data_dir / f"route_training_data_time.csv", route_time_trajs)
            else:
                logger.info(f"make stay trajectory by {time_threshold}min and {location_threshold}m")
                time_trajs, trajs = make_stay_trajectory(raw_trajs, time_threshold, location_threshold)
                logger.info("make complessed dataset by the grid")
                trajs, times, indice = make_complessed_dataset(time_trajs, trajs, grid)
                # send(training_data_dir / "indice.json")

                times = [[time[0] for time in traj] for traj in times]
            # route_dataset, route_times, _ = make_complessed_dataset(route_time_trajs, route_trajs, grid, indice)
            # route_times = [[time[0] for time in traj] for traj in route_times]
            # time_save_path = training_data_dir / f"route_training_data_time.csv"
            # logger.info(f"save route time dataset to {time_save_path}")
            # save(time_save_path, route_times)

            # save_path = training_data_dir / f"route_training_data.csv"
            # logger.info(f"save route complessed dataset to {save_path}")
            # save(save_path, route_dataset)

            gps = make_gps_data(training_data_dir, lat_range, lon_range, n_bins)
            make_distance_data(training_data_dir, n_bins, gps, logger)


        with open(training_data_dir / "indice.json", "w") as f:
            json.dump(indice, f)

        save_path = training_data_dir / f"training_data.csv"
        logger.info(f"save complessed dataset to {save_path}")
        save(save_path, trajs)
        
        time_save_path = training_data_dir / f"training_data_time.csv"
        logger.info(f"save time dataset to {time_save_path}")
        save(time_save_path, times)

        logger.info(f"saving setting to {training_data_dir}/params.json")
        with open(training_data_dir / "params.json", "w") as f:
            json.dump({"dataset": dataset_name, "n_locations": (n_bins+2)**2, "n_bins": n_bins, "seed": args.seed}, f)

    else:
        logger.info(f"training data already exists in {training_data_dir}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max_size', type=int)
    parser.add_argument('--n_bins', type=int)
    parser.add_argument('--time_threshold', type=int)
    parser.add_argument('--location_threshold', type=int)
    parser.add_argument('--truncate', type=int)
    parser.add_argument('--save_name', type=str)
    args = parser.parse_args()
    
    lat_range, lon_range = load_latlon_range(args.dataset)
    # n_bins = args.n_bins
    # training_data_dir = get_datadir() / args.dataset / args.data_name / args.save_name
    # training_data_dir.mkdir(exist_ok=True, parents=True)

    logger = set_logger(__name__, "./log.log")

    if args.dataset in ["peopleflow", "peopleflow_test"]:
        args.time_threshold = args.time_threshold / 60

    run(args.dataset, lat_range, lon_range, args.n_bins, args.time_threshold, args.location_threshold, args.max_size, args.seed, args.truncate, logger)