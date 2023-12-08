import argparse
from my_utils import get_datadir, save, load, construct_default_quadtree, set_logger
import pandas as pd
import numpy as np
import glob
import tqdm
from datetime import datetime
import os
import pathlib
import json
import linecache
import make_pair_to_route
import math

def make_raw_data_peopleflow():
    format = '%H:%M:%S'
    basic_time = datetime.strptime("00:00:00", format)
    def str_to_minute(time_str):
        return int((datetime.strptime(time_str, format) - basic_time).seconds / 60)

    for i in range(28):
        trajs = []
        peopleflow_raw_data_dir = f"/data/peopleflow/tokyo2008/p-csv/{i:04d}/*.csv"
        print("load raw data from", peopleflow_raw_data_dir)

        files = glob.glob(peopleflow_raw_data_dir)

        lat_index = 5
        lon_index = 4
        time_index = 3

        for file in tqdm.tqdm(files):
            trajectory = []
            df = pd.read_csv(file, header=None)
            time = 0
            for record in df.iterrows():
                record = record[1]
                lat, lon = float(record[lat_index]), float(record[lon_index])
                if time > str_to_minute(record[time_index].split(" ")[1]):
                    break
                else:
                    time = str_to_minute(record[time_index].split(" ")[1])
                trajectory.append((time, lat, lon))

            trajs.append(trajectory)
        
        save(save_path.parent.parent / "raw_data.csv", trajs, "a")

    np.random.seed(args.seed)
    trajs = load(save_path.parent.parent / "raw_data.csv", args.max_size)
    return trajs

def make_raw_data_distance_test(seed, max_size):

    np.random.seed(seed)
    possible_states = [0,1,2,3,4,5,6,7,8]


    P_r0 = [1/4,0,1/4,0,0,0,1/4,0,1/4]
    # P_r1 = [1/9]*9

    # P_r_r0 = [4/9,2/9,0,2/9,1/9,0,0,0,0]
    # P_r_r2 = [0,2/9,4/9,0,1/9,2/9,0,0,0]
    # P_r_r6 = [0,0,0,2/9,1/9,0,4/9,2/9,0]
    # P_r_r8 = [0,0,0,0,1/9,2/9,0,2/9,4/9]

    P_r_r0 = [0,2/5,0,2/5,1/5,0,0,0,0]
    P_r_r2 = [0,2/5,0,0,1/5,2/5,0,0,0]
    P_r_r6 = [0,0,0,2/5,1/5,0,0,2/5,0]
    P_r_r8 = [0,0,0,0,1/5,2/5,0,2/5,0]

    # sample r0 from P(r0)
    r0s = np.random.choice(possible_states, p=P_r0, size=max_size)
    # sample r1 from P(r1|r0)
    r1s = []
    for r0 in r0s:
        if r0 == 0:
            r1 = np.random.choice(possible_states, p=P_r_r0)
        elif r0 == 2:
            r1 = np.random.choice(possible_states, p=P_r_r2)
        elif r0 == 6:
            r1 = np.random.choice(possible_states, p=P_r_r6)
        elif r0 == 8:
            r1 = np.random.choice(possible_states, p=P_r_r8)
        else:
            raise ValueError("r0 is not in [0,2,6,8]")
        r1s.append(r1)

    r0s = np.array(r0s)
    r1s = np.array(r1s)

    # concat r0 and r1
    trajs = np.concatenate([r0s.reshape(-1,1), r1s.reshape(-1,1)], axis=1).tolist()

    data_name = "distance"
    data_dir = get_datadir() / "test" / data_name / f"seed{seed}_size{max_size}"
    save_path = data_dir / "training_data.csv"

    data_dir.mkdir(parents=True, exist_ok=True)
    save(save_path, trajs)

    times = []
    time_save_path = data_dir / "training_data_time.csv"
    for i in range(len(trajs)):
        times.append([(0,800), (800,1439)])
        
    max_time = 1439
    save(time_save_path, times, max_time)


def make_raw_data_test_circle(seed, max_size):

    np.random.seed(seed)

    # make data
    # the possible states are [1,2,3,4,5,6,7,8,9]
    # P(r0): [1/3,1/6,1/9,1/6,1/9,0,1/9,0,0]
    # P(r1): [0,0,1/9,0,1/9,1/6,1/9,1/6,1/3] if data_mode is normal
    # P(r1): [1,0,0,0,0,0,0,0,0] if data_mode is simple

    possible_states = [0,1,2,3,4,5,6,7,8]

    P_r0 = np.zeros(9)
    P_r0[[0,2,6,8]] = 9/40
    P_r0[[1,3,4,5,7]] = 1/50

    # P(r1|r0=0): [0,0,1/3,0,1/3,0,1/3,0,0]
    # P(r1|r0=1,3): [0,0,0,0,0,1/2,0,1/2,0]
    # P(r1|r0=2,4,6): [0,0,0,0,0,0,0,0,1]

    P_r_0_1 = np.zeros(9)
    P_r_3_6 = np.zeros(9)
    P_r_7_8 = np.zeros(9)
    P_r_2_5 = np.zeros(9)
    P_r_4 = np.zeros(9)
    P_r_0_1[[3,6]] = 1/2
    P_r_3_6[[7,8]] = 1/2
    P_r_7_8[[2,5]] = 1/2
    P_r_2_5[[0,1]] = 1/2
    P_r_4[[0,1,2,3,5,6,7,8]] = 1/8

    # sample r0 from P(r0)
    r0s = np.random.choice(possible_states, p=P_r0, size=max_size)
    # sample r1 from P(r1|r0)
    r1s = []
    for r0 in r0s:
        if r0 == 0 or r0 == 1:
            r1 = np.random.choice(possible_states, p=P_r_0_1)
        elif r0 == 3 or r0 == 6:
            r1 = np.random.choice(possible_states, p=P_r_3_6)
        elif r0 == 7 or r0 == 8:
            r1 = np.random.choice(possible_states, p=P_r_7_8)
        elif r0 == 2 or r0 == 5:
            r1 = np.random.choice(possible_states, p=P_r_2_5)
        elif r0 == 4:
            r1 = np.random.choice(possible_states, p=P_r_4)
        else:
            raise ValueError("r0 is not in [0,1,2,3]")
        r1s.append(r1)

    r0s = np.array(r0s)
    r1s = np.array(r1s)

    # concat r0 and r1
    trajs = np.concatenate([r0s.reshape(-1,1), r1s.reshape(-1,1)], axis=1).tolist()

    data_name = "circle"
    data_dir = get_datadir() / "test" / data_name / f"seed{seed}_size{max_size}"
    save_path = data_dir / "training_data.csv"

    data_dir.mkdir(parents=True, exist_ok=True)
    save(save_path, trajs)

    times = []
    time_save_path = data_dir / "training_data_time.csv"
    for i in range(len(trajs)):
        if len(trajs[i]) == 2:
            times.append([(0,800), (800,1439)])
        else:
            times.append([(0,800), (800,1200), (1200,1439)])
        
    max_time = 1439
    save(time_save_path, times, max_time)

    json_file = {"lat_range": [34.95, 36.85], "lon_range": [138.85, 140.9], "start_hour": 0, "end_hour": 23, "n_bins": n_bins, "save_name": data_name, "dataset": "test"}
    with open(data_dir / "params.json", "w") as f:
        json.dump(json_file, f)
    

def make_raw_data_test(seed, max_size, mode, is_variable, n_bins):

    np.random.seed(seed)

    # make data
    # the possible states are [1,2,3,4,5,6,7,8,9]
    # P(r0): [1/3,1/6,1/9,1/6,1/9,0,1/9,0,0]
    # P(r1): [0,0,1/9,0,1/9,1/6,1/9,1/6,1/3] if data_mode is normal
    # P(r1): [1,0,0,0,0,0,0,0,0] if data_mode is simple

    possible_states = [0,1,2,3,4,5,6,7,8]

    if mode == "normal":
        P_r0 = [1/3,1/6,1/9,1/6,1/9,0,1/9,0,0]
    elif mode == "simple":
        P_r0 = [1,0,0,0,0,0,0,0,0]
    P_r1 = [0,0,1/9,0,1/9,1/6,1/9,1/6,1/3]

    # P(r1|r0=0): [0,0,1/3,0,1/3,0,1/3,0,0]
    # P(r1|r0=1,3): [0,0,0,0,0,1/2,0,1/2,0]
    # P(r1|r0=2,4,6): [0,0,0,0,0,0,0,0,1]

    P_r1_r01 = [0,0,1/3,0,1/3,0,1/3,0,0]
    P_r1_r02 = [0,0,0,0,0,1/2,0,1/2,0]
    P_r1_r03 = [0,0,0,0,0,0,0,0,1]

    # sample r0 from P(r0)
    r0s = np.random.choice(possible_states, p=P_r0, size=max_size)
    # sample r1 from P(r1|r0)
    r1s = []
    for r0 in r0s:
        if r0 == 0:
            r1 = np.random.choice(possible_states, p=P_r1_r01)
        elif r0 == 1 or r0 == 3:
            r1 = np.random.choice(possible_states, p=P_r1_r02)
        elif r0 == 2 or r0 == 4 or r0 == 6:
            r1 = np.random.choice(possible_states, p=P_r1_r03)
        else:
            raise ValueError("r0 is not in [0,1,2,3]")
        r1s.append(r1)

    r0s = np.array(r0s)
    r1s = np.array(r1s)

    # concat r0 and r1
    trajs = np.concatenate([r0s.reshape(-1,1), r1s.reshape(-1,1)], axis=1).tolist()

    if is_variable:
        # if r0 is 0 and r1 is 2, add 8
        for i in range(len(trajs)):
            if trajs[i][0] == 0 and trajs[i][1] == 2:
                trajs[i].extend(np.random.choice(possible_states, size=1).tolist())

    if is_variable:
        data_name = f"{mode}_variable"
    else:
        data_name = f"{mode}"

    data_dir = get_datadir() / "test" / data_name / f"seed{seed}_size{max_size}_nbins{n_bins}"
    save_path = data_dir / "training_data.csv"

    data_dir.mkdir(parents=True, exist_ok=True)
    save(save_path, trajs)

    times = []
    time_save_path = data_dir / "training_data_time.csv"
    for i in range(len(trajs)):
        if len(trajs[i]) == 2:
            times.append([(0,800), (800,1439)])
        else:
            times.append([(0,800), (800,1200), (1200,1439)])
        
    max_time = 1439
    save_time_with_nan_padding(time_save_path, times, max_time)

    json_file = {"lat_range": [34.95, 36.85], "lon_range": [138.85, 140.9], "start_hour": 0, "end_hour": 23, "n_bins": n_bins, "save_name": data_name, "dataset": "test"}
    with open(data_dir / "params.json", "w") as f:
        json.dump(json_file, f)
    

def make_raw_data_rotation(seed, max_size, n_bins):
    # data_dir = get_datadir() / "rotation" / str(max_size) / f"bin{n_bins}_seed{seed}"
    # save_path = data_dir / "training_data.csv"

    # if save_path.exists():
        # print("already exists")
        # return
    
    depth = 2
    assert n_bins >= 6
    np.random.seed(seed)
    tree = construct_default_quadtree(n_bins)
    tree.make_self_complete()
    states = tree.root_node.state_list
    candidates_for_the_start_locations = []
    for state in states:
        location_id_in_the_depth = tree.get_location_id_in_the_depth(state, depth)
        if (location_id_in_the_depth % 2 == 0) and (location_id_in_the_depth < 4**1 + 4**(2)-2):
            candidates_for_the_start_locations.append(state)
    
    trajs = []
    for i in range(max_size):
        start_location = np.random.choice(candidates_for_the_start_locations)
        location_id_in_the_depth = tree.get_location_id_in_the_depth(start_location, depth)
        mediate_location_id_in_the_depth = location_id_in_the_depth + 1
        mediate_node_id = tree.node_id_to_hidden_id.index(mediate_location_id_in_the_depth)
        mediate_node = tree.get_node_by_id(mediate_node_id)
        mediate_location = np.random.choice(mediate_node.state_list)
        end_location_id_in_the_depth = mediate_location_id_in_the_depth + 1
        end_node_id = tree.node_id_to_hidden_id.index(end_location_id_in_the_depth)
        end_node = tree.get_node_by_id(end_node_id)
        end_location = np.random.choice(end_node.state_list)
        trajs.append([start_location, mediate_location, end_location])


    # data_dir.mkdir(parents=True, exist_ok=True)
    
    # print("save to", save_path)
    # save(save_path, trajs)

    times = []
    # time_save_path = data_dir / "training_data_time.csv"
    times = [[0,1,2]]*len(trajs)
    # print("save to", time_save_path)
    # save(time_save_path, times)

    return trajs, times


def make_raw_data_random(seed, max_size, n_bins):
    # data_dir = get_datadir() / "random" / str(max_size) / f"bin{n_bins}_seed{seed}"
    # save_path = data_dir / "training_data.csv"

    # if save_path.exists():
        # print("already exists")
        # return
    # else:
        # save_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_candidate_locations = 1
    np.random.seed(seed)

    n_locations = (n_bins+2)**2
    candidates_for_the_start_locations = list(range(n_locations))
    candidates_list_for_the_end_locations = [np.random.choice(range(n_locations), size=n_candidate_locations) for i in candidates_for_the_start_locations]
    trajs = []
    for i in range(max_size):
        start_location = np.random.choice(candidates_for_the_start_locations)
        candidates_for_the_end_locations = candidates_list_for_the_end_locations[start_location]
        end_location = np.random.choice(candidates_for_the_end_locations)
        trajs.append([start_location, end_location])

    # print("save to", save_path)
    # save(save_path, trajs)

    times = []
    # time_save_path = data_dir / "training_data_time.csv"
    times = [[0,1]]*len(trajs)
    # print("save to", time_save_path)
    # save(time_save_path, times)
    return trajs, times

def make_raw_data_geolife(test=False):
    # the format of plt_file: records follow after the 6 raws for auxiliary information
    # the format of a record: lat, lon, unused, altitude, days since dec. 30 1899, date, time
    # https://heremaps.github.io/pptk/tutorials/viewer/geolife.html

    if test:
        data_name = "geolife_test"
    else:
        data_name = "geolife"
    # define the directory path
    dir_path = os.path.join(f"temp/Geolife Trajectories 1.3/Data/", "{:03}/Trajectory/")
    plt_files = []

    # loop through all the directories and files
    for i in range(181):
        path = dir_path.format(i)
        if os.path.exists(path):
            files = os.listdir(path)
            files = [os.path.join(path, f) for f in files if f.endswith('.plt')]
            plt_files.extend(files)
        else:
            print(f'{path} does not exist')
            
        if test and i == 0:
            break

    def read_plt_file(plt_file):
        with open(plt_file, 'r') as f:
            # skip the first 6 lines
            for i in range(6):
                next(f)

            # read the remaining lines
            traj = []
            base = 0
            total_minutes = 0
            for i, line in enumerate(f):
                lat, lon, _, _, _, _, time = line.strip().split(',')
                lat, lon = float(lat), float(lon)
                hours, minutes, seconds = map(int, time.split(':'))
                # when the time is smaller than the previous time, it means the next day
                if total_minutes > base + (hours * 60 + minutes) * 60 + seconds:
                    base += 1439 * 60
                total_minutes = base + (hours * 60 + minutes) * 60 + seconds

                if not (-90 < lat < 90 and -180 < lon < 180):
                    continue

                traj.append((total_minutes, lat, lon))

        return traj

    # save_path = "/data/geolife/raw_data.csv"
    # with open(save_path, 'w') as f:
    trajs = []
    for plt_file in plt_files:
        traj = read_plt_file(plt_file)
        trajs.append(traj)
        # traj is written in the format of "total_minutes lat lon,total_minutes lat lon,..."
        # f.write(','.join(map(lambda x: ' '.join(map(str, x)), traj)) + '\n')

    # trajs = load(save_path, args.max_size, args.seed)
    return trajs

def make_raw_data_test_quadtree(seed, max_size, n_bins):

    np.random.seed(seed)
    n_locations = (n_bins+2)**2
    possible_states = list(range(n_locations))
    P_r0 = np.zeros(n_locations)
    P_r0[3] = 0.5
    P_r0[[8,9,12,13,15]] = 0.1
    P_r1 = np.zeros(n_locations)
    P_r1[0] = 1

    # sample r0 from P(r0)
    r0s = np.random.choice(possible_states, p=P_r0, size=max_size)
    # sample r1 from P(r1|r0)
    r1s = []
    for r0 in r0s:
        r1 = np.random.choice(possible_states, p=P_r1)
        r1s.append(r1)

    r0s = np.array(r0s)
    r1s = np.array(r1s)

    # concat r0 and r1
    trajs = np.concatenate([r0s.reshape(-1,1), r1s.reshape(-1,1)], axis=1).tolist()

    for i in range(len(trajs)):
        if trajs[i][0] == 3:
            trajs[i].extend([3])

    data_name = "quadtree"
    data_dir = get_datadir() / "test" / data_name / f"seed{seed}_size{max_size}_nbins{n_bins}"
    save_path = data_dir / "training_data.csv"

    data_dir.mkdir(parents=True, exist_ok=True)
    save(save_path, trajs)

    times = []
    time_save_path = data_dir / "training_data_time.csv"
    for i in range(len(trajs)):
        if len(trajs[i]) == 2:
            times.append([0,1])
        else:
            times.append([0,1,2])
        
    save(time_save_path, times)

    json_file = {"lat_range": [34.95, 36.85], "lon_range": [138.85, 140.9], "start_hour": 0, "end_hour": 23, "n_bins": n_bins, "save_name": data_name, "dataset": "test"}
    with open(data_dir / "params.json", "w") as f:
        json.dump(json_file, f)


def make_raw_data_test_return(seed, max_size):

    np.random.seed(seed)

    # make data
    # the possible states are [1,2,3,4,5,6,7,8,9]
    # P(r0): [1/3,1/6,1/9,1/6,1/9,0,1/9,0,0]
    # P(r1): [0,0,1/9,0,1/9,1/6,1/9,1/6,1/3] if data_mode is normal
    # P(r1): [1,0,0,0,0,0,0,0,0] if data_mode is simple

    possible_states = [0,1,2,3,4,5,6,7,8]
    P_r0 = [1/3,1/6,1/9,1/6,1/9,0,1/9,0,0]
    P_r1 = [0,0,1/9,0,1/9,1/6,1/9,1/6,1/3]

    # P(r1|r0=0): [0,0,1/3,0,1/3,0,1/3,0,0]
    # P(r1|r0=1,3): [0,0,0,0,0,1/2,0,1/2,0]
    # P(r1|r0=2,4,6): [0,0,0,0,0,0,0,0,1]

    P_r1_r01 = [0,0,1/3,0,1/3,0,1/3,0,0]
    P_r1_r02 = [0,0,0,0,0,1/2,0,1/2,0]
    P_r1_r03 = [0,0,0,0,0,0,0,0,1]

    # sample r0 from P(r0)
    r0s = np.random.choice(possible_states, p=P_r0, size=max_size)
    # sample r1 from P(r1|r0)
    r1s = []
    for r0 in r0s:
        if r0 == 0:
            r1 = np.random.choice(possible_states, p=P_r1_r01)
        elif r0 == 1 or r0 == 3:
            r1 = np.random.choice(possible_states, p=P_r1_r02)
        elif r0 == 2 or r0 == 4 or r0 == 6:
            r1 = np.random.choice(possible_states, p=P_r1_r03)
        else:
            raise ValueError("r0 is not in [0,1,2,3]")
        r1s.append(r1)

    r0s = np.array(r0s)
    r1s = np.array(r1s)

    # concat r0 and r1
    trajs = np.concatenate([r0s.reshape(-1,1), r1s.reshape(-1,1)], axis=1).tolist()

    # if r0 is 0 and r1 is 2, add 0
    for i in range(len(trajs)):
        if trajs[i][0] == 0 and trajs[i][1] == 2:
            trajs[i].extend([0])

    data_name = "return"
    data_dir = get_datadir() / "test" / data_name / f"seed{seed}_size{max_size}"
    save_path = data_dir / "training_data.csv"

    data_dir.mkdir(parents=True, exist_ok=True)
    save(save_path, trajs)

    times = []
    time_save_path = data_dir / "training_data_time.csv"
    for i in range(len(trajs)):
        if len(trajs[i]) == 2:
            times.append([(0,800), (800,1439)])
        else:
            times.append([(0,800), (800,1200), (1200,1439)])
        
    max_time = 1439
    save_time_with_nan_padding(time_save_path, times, max_time)

    json_file = {"lat_range": [34.95, 36.85], "lon_range": [138.85, 140.9], "start_hour": 0, "end_hour": 23, "n_bins": n_bins, "save_name": data_name, "dataset": "test"}
    with open(data_dir / "params.json", "w") as f:
        json.dump(json_file, f)

def make_raw_data_taxi():

    original_data_path = '/data/taxi_raw/raw/train.csv'
    print("load raw data from", original_data_path)
    df = pd.read_csv(original_data_path)

    # remove the record with missing data
    df = df[df['MISSING_DATA'] == False]

    # convert the trajectory data to a list of points
    # the trajectory data is string of the form "[[x1,y1],[x2,y2],...,[xn,yn]]"
    # the list of points is a list of tuples (x,y)
    df["POLYLINE"] = df["POLYLINE"].apply(lambda x: eval(x))

    # convert the list to the format
    # [[lon,lat],...] -> [[time,lat,lon],...]
    # the time starts from 0 and the unit is minute
    def convert_to_list_of_points(polyline):
        if polyline == []:
            return []
        else:
            return [[i,point[1],point[0]] for i,point in enumerate(polyline)]

    trajs = []
    for i in df.index:
        traj = convert_to_list_of_points(df["POLYLINE"][i])
        if traj != []:
            trajs.append(traj)

    return trajs


def process_map_matched_data(data_dir, trajs):
    """
    map matched data is a sequence of edges
    we convert the sequence of edges to a sequence of points
    That is, we convert an edge to a point (the start point) and apply it to the all edges
    """
    nodes_edges = make_pair_to_route.load_edges(data_dir)

    new_trajs = []
    for traj in trajs:
        # edge_id = traj[0]
        # new_traj = [nodes_edges[edge_id][0], nodes_edges[edge_id][1]]
        new_traj = []
        for edge in traj:
            new_traj.append(nodes_edges[edge][0])
        new_trajs.append(new_traj)
    return new_trajs, nodes_edges

def load_map_matched_data(data_dir):
    """
    return a list of map matched trajs
    a traj is a sequence of edges
    """

    data_path = pathlib.Path(data_dir) / "training_data.csv"
    original_trajs = []

    with open(data_path, "r") as f:
        for line in f:
            traj = [int(x) for x in line.split()][:-1]
            original_trajs.append([v for v in traj])
    
    data_path = pathlib.Path(data_dir) / "training_data_time.csv"
    original_time_trajs = []

    with open(data_path, "r") as f:
        for line in f:
            # print(line)
            traj = [float(x)/60 for x in line.split()]
            # if it's nan it will be converted to 0
            traj = [0 if math.isnan(x) else int(x) for x in traj]
            traj[0] = 0
            original_time_trajs.append(traj)

    return original_trajs, original_time_trajs

def make_raw_data_from_map_matched_data(data_dir):

    original_trajs, original_time_trajs = load_map_matched_data(data_dir)
    processed_trajs, _ = process_map_matched_data(data_dir, original_trajs)

    trajs = []
    for i in range(len(processed_trajs)):
        traj = processed_trajs[i]
        time_traj = original_time_trajs[i] + [0]
        new_traj = []
        # print(traj, time_traj)
        for j in range(len(traj)):
            new_traj.append([time_traj[j], traj[j][0], traj[j][1]])
        trajs.append(new_traj)
        

    return trajs


def make_raw_data(dataset, logger):
    """
    save_path: the path to save the raw data (pathlib)
    raw_data: sequence of (time, lat, lon)
    """
    save_path = get_datadir() / dataset / "raw_data.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not (save_path).exists():
        if dataset == "geolife":
            trajs = make_raw_data_geolife()
        elif dataset == "geolife_mm":
            convert_mr_to_training(dataset)
            trajs = make_raw_data_from_map_matched_data(get_datadir() / dataset / "raw")
        elif dataset == "geolife_test":
            trajs = make_raw_data_geolife(True)
        elif dataset == "geolife_test_mm":
            logger.info(f"make raw data from map matched data {save_path.parent / 'raw'}")
            convert_mr_to_training(dataset)
            trajs = make_raw_data_from_map_matched_data(get_datadir() / dataset / "raw")
        elif dataset == "chengdu":
            logger.info(f"make raw data from map matched data {save_path.parent / 'raw'}")
            trajs = make_raw_data_from_map_matched_data(save_path.parent / "raw")
        else:
            print("skip make_raw_data", dataset)
            trajs = None

        if trajs is not None:
            save(save_path, trajs)
    #     if original_data_name == 'taxi':
    #         trajs = make_raw_data_taxi()
    #     elif original_data_name == 'peopleflow':
    #         trajs = make_raw_data_peopleflow()
    #         save_path.parent.parent.mkdir(parents=True, exist_ok=True)
    #         print("save raw data to", save_path.parent.parent / "raw_data.csv")
    #         save(save_path.parent.parent / "raw_data.csv", trajs)
    #     elif original_data_name == "geolife":
    #         trajs = make_raw_data_geolife()
    #     elif original_data_name == "geolife_mm":
    #         trajs = make_raw_data_from_map_matched_data(save_path.parent / "MTNet")
    #     elif original_data_name == "chengdu":
    #         logger.info(f"make raw data from map matched data {save_path.parent.parent / 'raw'}")
    #         trajs = make_raw_data_from_map_matched_data(save_path.parent.parent / "raw")
    #         # data_path = "dataset_configs/chengdu.json"
    #         # with open(data_path, "w") as f:
    #             # json.dump({"lat_range": lat_range, "lon_range": lon_range}, f)
    #     elif original_data_name == 'circle':
    #         trajs = make_raw_data_test_circle(seed, max_size) 
    #     elif original_data_name == 'return':
    #         trajs = make_raw_data_test_return(seed, max_size)
    #     elif original_data_name == 'quadtree':
    #         trajs = make_raw_data_test_quadtree(seed, max_size, n_bins)
    #     elif original_data_name == 'rotation':
    #         make_raw_data_rotation(seed, max_size, n_bins)
    #     elif original_data_name == 'random':
    #         make_raw_data_random(seed, max_size, n_bins)
    #     else:
    #         trajs = make_raw_data_test(args.seed, args.max_size, "normal", True, args.n_bins)
    #         trajs = make_raw_data_distance_test(args.seed, args.max_size)
    else:
        print("skip because raw data already exists in", save_path)
        # np.random.seed(args.seed)
        # trajs = load(save_path.parent.parent / "raw_data.csv", args.max_size)
    # return trajs


def convert_mr_to_training(dataset):
    assert dataset.split("_")[-1] == "mm", "dataset must be map matched"

    # dataset = "_".join(dataset.split("_")[:-1])
    data_dir = get_datadir() / dataset / "raw"
    save_dir = get_datadir() / dataset / "raw"

    max_length = 1000
    # send mr to the server for backup
    # send(os.path.join(data_dir, "mr.txt"))
    # format of training: edge_id edge_id ... edge_id 0

    # load times
    with open(os.path.join(data_dir, "times.csv"), "r") as f:
        f.readline()
        times = []
        for line in f:
            time = line.split(",")
            time = [float(t) for t in time if t != ""]
            times.append(time)

    training_data = []
    training_data_time = []
    n_strange = 0
    with open(os.path.join(data_dir, "mr.txt"), "r") as f:
        f.readline()
        for line in f:
            id = int(line.split(";")[0])
            edge_ids_for_each_point = line.split(";")[1]
            edge_ids = line.split(";")[2]
            wkt = line.split(";")[3]

            edge_ids = edge_ids.split(",")
            # convert to int
            edge_ids = [int(edge_id) for edge_id in edge_ids if edge_id != ""]
            # if it includes 0, it means that map matching failed
            if len(edge_ids) == 0:
                continue
            # edge_ids.append(0)

            edge_ids_for_each_point = edge_ids_for_each_point.split(",")
            # convert to int
            edge_ids_for_each_point = [int(edge_id) for edge_id in edge_ids_for_each_point if edge_id != ""]

            assert len(times[id-1]) == len(edge_ids_for_each_point), f"{len(times[id-1])} != {len(edge_ids_for_each_point)}"

            # get the indice that change the edge
            change_edge_indices = [0] + [i+1 for i in range(len(edge_ids_for_each_point)-1) if edge_ids_for_each_point[i] != edge_ids_for_each_point[i+1]]
            edge_ids_ = [edge_ids_for_each_point[i] for i in change_edge_indices] + [0]
            # get the time of the change
            change_times = [times[id-1][i] for i in change_edge_indices]
            # get the difference of the time
            change_times = [0] + [int(change_times[i+1]-change_times[i]) for i in range(len(change_times)-1)]

            # edge_ids_ <- original edges
            # edge_ids <- connected by compensation if two adjacent edges are not connected
            # add 0 to the times where the edge is compensated
            cursor = 0
            for i in range(len(edge_ids_)-1):
                current_edge = edge_ids_[i]
                while current_edge != edge_ids[cursor]:
                    cursor += 1
                    change_times.insert(cursor, 0)
                cursor += 1

            if len(change_times) != len(edge_ids):
                n_strange += 1
                print("WARNING: diffenrt length", len(change_times), len(edge_ids), n_strange)
                print(edge_ids, edge_ids_)
                edge_ids = edge_ids[:len(change_times)]
            # if len(edge_ids_) != len(edge_ids)+1:
                # print("skip because an edge is not connected")
                # continue

            training_data_time.append(change_times[:max_length])
            training_data.append(edge_ids[:max_length] + [0])
            # training_data.append(edge_ids_)
            assert len(change_times) == len(edge_ids), f"{len(change_times)} != {len(edge_ids)}"
    
    with open(os.path.join(save_dir, "training_data.csv"), "w") as f:
        for edge_ids in training_data:
            f.write(" ".join([str(edge_id) for edge_id in edge_ids])+"\n")
    
    with open(os.path.join(save_dir, "training_data_time.csv"), "w") as f:
        for times in training_data_time:
            f.write(" ".join([str(time) for time in times])+"\n")

    # send(os.path.join(save_dir, "training_data.csv"))
    # send(os.path.join(save_dir, "training_data_time.csv"))


def run(dataset):
    logger = set_logger(__name__, "./log.log")
    # target to make
    # save_path = get_datadir() / original_data_name / save_name / f"raw_data_seed{seed}.csv"
    # if save_path.exists():
        # logger.info(f"raw data already exists in {save_path}")
        # return
    
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # make raw data
    make_raw_data(dataset, logger)

    # load raw data
    # trajs = load(save_path, max_size)

    # np.random.seed(seed)
    # if args.max_size != 0:
    #     print("shuffle trajectories and choose the first", args.max_size, "trajectories")
    #     # shuffle trajectories and real_time_traj with the same order without using numpy
    #     p = np.random.permutation(len(trajs))
    #     trajs = [trajs[i] for i in p]
    #     trajs = trajs[:max_size]

    # logger.info(f"save raw data to {save_path}")
    # save(save_path, trajs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    run(args.dataset)

    # save_path = get_datadir() / args.original_data_name / args.save_name / f"raw_data_seed{args.seed}.csv"
    # save_paths = glob.glob(str(save_path))
    # if save_paths != []:
    #     print("raw data already exists")
    #     exit()
    
    # trajs = None
    # if not (save_path.parent.parent / "raw_data.csv").exists():
    #     print("make raw data")
    #     if args.original_data_name == 'taxi':
    #         trajs = make_raw_data_taxi()
    #     elif args.original_data_name == 'peopleflow':
    #         trajs = make_raw_data_peopleflow()
    #         save_path.parent.parent.mkdir(parents=True, exist_ok=True)
    #         print("save raw data to", save_path.parent.parent / "raw_data.csv")
    #         save(save_path.parent.parent / "raw_data.csv", trajs)
    #     elif args.original_data_name == "geolife":
    #         trajs = make_raw_data_geolife()
    #     elif args.original_data_name == "geolife_mm":
    #         trajs = make_raw_data_from_map_matched_data(save_path.parent / "MTNet")
    #     elif args.original_data_name == "chengdu":
    #         trajs = make_raw_data_from_map_matched_data(save_path.parent / "MTNet")
    #         # data_path = "dataset_configs/chengdu.json"
    #         # with open(data_path, "w") as f:
    #             # json.dump({"lat_range": lat_range, "lon_range": lon_range}, f)
    #     elif args.original_data_name == 'circle':
    #         trajs = make_raw_data_test_circle(args.seed, args.max_size) 
    #     elif args.original_data_name == 'return':
    #         trajs = make_raw_data_test_return(args.seed, args.max_size)
    #     elif args.original_data_name == 'quadtree':
    #         trajs = make_raw_data_test_quadtree(args.seed, args.max_size, args.n_bins)
    #     elif args.original_data_name == 'rotation':
    #         make_raw_data_rotation(args.seed, args.max_size, args.n_bins)
    #     elif args.original_data_name == 'random':
    #         make_raw_data_random(args.seed, args.max_size, args.n_bins)
    #     else:
    #         trajs = make_raw_data_test(args.seed, args.max_size, "normal", True, args.n_bins)
    #         trajs = make_raw_data_distance_test(args.seed, args.max_size)
    # else:
    #     print("load raw data from", save_path.parent.parent / "raw_data.csv", args.max_size, "data")
    #     np.random.seed(args.seed)
    #     trajs = load(save_path.parent.parent / "raw_data.csv", args.max_size)

    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # if trajs is not None:

    #     np.random.seed(args.seed)
    #     if args.max_size != 0:
    #         print("shuffle trajectories and choose the first", args.max_size, "trajectories")
    #         # shuffle trajectories and real_time_traj with the same order without using numpy
    #         p = np.random.permutation(len(trajs))
    #         trajs = [trajs[i] for i in p]
    #         trajs = trajs[:args.max_size]

    #     print("save raw data to", save_path)
    #     save(save_path, trajs)