import argparse
from my_utils import get_datadir
import pandas as pd
from data_pre_processing import save_timelatlon_with_nan_padding, save_state_with_nan_padding, save_time_with_nan_padding, compute_distance_matrix
import numpy as np
import glob
import tqdm
from datetime import datetime
import os
import pathlib
import json

    
def make_raw_data_peopleflow():
    format = '%H:%M:%S'
    basic_time = datetime.strptime("00:00:00", format)
    def str_to_minute(time_str):
        return int((datetime.strptime(time_str, format) - basic_time).seconds / 60)

    trajs = []
    for i in range(28):
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
    save_state_with_nan_padding(save_path, trajs)

    times = []
    time_save_path = data_dir / "training_data_time.csv"
    for i in range(len(trajs)):
        times.append([(0,800), (800,1439)])
        
    max_time = 1439
    save_time_with_nan_padding(time_save_path, times, max_time)


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
    save_state_with_nan_padding(save_path, trajs)

    times = []
    time_save_path = data_dir / "training_data_time.csv"
    for i in range(len(trajs)):
        if len(trajs[i]) == 2:
            times.append([(0,800), (800,1439)])
        else:
            times.append([(0,800), (800,1200), (1200,1439)])
        
    max_time = 1439
    save_time_with_nan_padding(time_save_path, times, max_time)

    json_file = {"lat_range": [34.95, 36.85], "lon_range": [138.85, 140.9], "start_hour": 0, "end_hour": 23, "n_bins": 1, "save_name": data_name, "dataset": "test"}
    with open(data_dir / "params.json", "w") as f:
        json.dump(json_file, f)
    

def make_raw_data_test(seed, max_size, mode, is_variable):

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

    data_dir = get_datadir() / "test" / data_name / f"seed{seed}_size{max_size}"
    save_path = data_dir / "training_data.csv"

    data_dir.mkdir(parents=True, exist_ok=True)
    save_state_with_nan_padding(save_path, trajs)

    times = []
    time_save_path = data_dir / "training_data_time.csv"
    for i in range(len(trajs)):
        if len(trajs[i]) == 2:
            times.append([(0,800), (800,1439)])
        else:
            times.append([(0,800), (800,1200), (1200,1439)])
        
    max_time = 1439
    save_time_with_nan_padding(time_save_path, times, max_time)

    json_file = {"lat_range": [34.95, 36.85], "lon_range": [138.85, 140.9], "start_hour": 0, "end_hour": 23, "n_bins": 1, "save_name": data_name, "dataset": "test"}
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
    save_state_with_nan_padding(save_path, trajs)

    times = []
    time_save_path = data_dir / "training_data_time.csv"
    for i in range(len(trajs)):
        if len(trajs[i]) == 2:
            times.append([(0,800), (800,1439)])
        else:
            times.append([(0,800), (800,1200), (1200,1439)])
        
    max_time = 1439
    save_time_with_nan_padding(time_save_path, times, max_time)

    json_file = {"lat_range": [34.95, 36.85], "lon_range": [138.85, 140.9], "start_hour": 0, "end_hour": 23, "n_bins": 1, "save_name": data_name, "dataset": "test"}
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

def make_raw_data_chengdu():
    data_path = "/data/chengdu/raw/trajs_demo.csv"
    trajs = []
    original_trajs = []
    with open(data_path, "r") as f:
        for line in f:
            traj = [int(x) for x in line.split()][:-1]
            trajs.append([traj[0]-1, traj[-1]-1])
            original_trajs.append([v-1 for v in traj])
            
    return trajs, original_trajs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_data_name', type=str)
    parser.add_argument('--max_size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--save_name', type=str)
    args = parser.parse_args()


    save_path = get_datadir() / args.original_data_name / args.save_name / "raw_data*.csv"
    save_paths = glob.glob(str(save_path))
    if save_paths != []:
        print("raw data already exists")
        exit()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if args.original_data_name == 'taxi':
        trajs = make_raw_data_taxi()
    elif args.original_data_name == 'peopleflow':
        trajs = make_raw_data_peopleflow()
    elif args.original_data_name == "chengdu":
        trajs, original_trajs = make_raw_data_chengdu()
    elif args.original_data_name == 'test':
        print("make raw data")
        if args.save_name == 'circle':
            trajs = make_raw_data_test_circle(args.seed, args.max_size) 
        elif args.save_name == 'return':
            trajs = make_raw_data_test_return(args.seed, args.max_size)
        else:
            trajs = make_raw_data_test(args.seed, args.max_size, "normal", True)
            trajs = make_raw_data_distance_test(args.seed, args.max_size)


    if trajs is not None:
        np.random.seed(args.seed)
        if args.max_size != 0:
            # shuffle trajectories and real_time_traj with the same order without using numpy
            p = np.random.permutation(len(trajs))
            trajs = [trajs[i] for i in p]
            trajs = trajs[:args.max_size]

        if args.original_data_name == 'chengdu':
            original_trajs = [original_trajs[i] for i in p]
            original_trajs = original_trajs[:args.max_size]
            save_state_with_nan_padding(save_path.parent / "original_data.csv", original_trajs)

            time_trajs = [[0,800,1439]] * len(trajs)
            save_dir = save_path.parent / "start_end"
            save_dir.mkdir(parents=True, exist_ok=True)
            print("save raw data to", save_dir / "training_data.csv")
            save_state_with_nan_padding(save_dir / "training_data.csv", trajs)
            print("save raw time data to", save_dir / "training_data_time.csv")
            save_state_with_nan_padding(save_dir / "training_data_time.csv", time_trajs)
            data_path = "/data/chengdu/raw/edge_property.txt"
            latlons = []
            with open(data_path) as f:
                for line in f:
                    raw = line.split("LINESTRING")[1][1:-3].split(",")
                    lon, lat = np.average([list(map(float,latlon.split())) for latlon in raw], axis=0)
                    latlons.append([lat, lon])
            gps = pd.DataFrame(latlons)
            gps.to_csv(save_dir / f"gps.csv", header=None, index=None)
            if not pathlib.Path("/data/chengdu/1000/start_end/distance_matrix.npy").exists():
                print("compute distance matrix and save to", save_dir / "distance_matrix.npy")
                def get_latlon(state):
                    return gps.iloc[state].tolist()
                distance_matrix = compute_distance_matrix(get_latlon, len(gps))
                np.save(save_dir/f'distance_matrix.npy',distance_matrix)
            else:
                # copy /data/chengdu/1000/start_end/distance_matrix.npy to save_dir / "distance_matrix.npy"
                print("copy distance matrix from /data/chengdu/1000/start_end/distance_matrix.npy to", save_dir / "distance_matrix.npy")
                os.system(f"cp /data/chengdu/1000/start_end/distance_matrix.npy {save_dir}/distance_matrix.npy")
        else:
            print("save raw data to", save_path)
            save_timelatlon_with_nan_padding(save_path, trajs)
