# data format
# #0:
# >0:5.732,5.232;5.731,5.219;
# #1:
# >0:5.785,5.230;5.767,5.227;5.794,5.256;
# #2:
# >0:5.840,5.272;5.832,5.268;
# .....

# Above is an example of a file of trajectories.
# Every trajectory data contains two lines. The first line shows the index of the trajectory and the second line is the location points of the trajectory.
# The format of the first line is "#i:", where i is the index of the trajectory.
# The second line starts with a sign ">0:" that does not have any meaning. Following are the location points of the trajectory. Every point has the format "x,y" and every two points are divided by a ";".

# The following code is to read the data and convert it to the format
import pandas as pd
import json
import argparse
import pathlib

import sys
sys.path.append("../../")
from my_utils import load, get_original_dataset_name
from grid import Grid
from data_pre_processing import check_in_range

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--training_data_name', type=str)
    parser.add_argument('--dataset_seed', type=int)
    parser.add_argument('--dataset_config_path', type=str)
    args = parser.parse_args()

    path = pathlib.Path(f"/data/{args.dataset}/{args.data_name}/{args.training_data_name}")
    # if path.exists():
    #     print(f"file {path} already exists")
    #     exit(0)

    with open(path / "params.json", "r") as f:
        params = json.load(f)
    n_bins = params["n_bins"]

    dataset_name = get_original_dataset_name(args.dataset)
    # original_data_path = pathlib.Path(f"/data/{args.dataset}/{args.data_name}/raw_data_seed{args.dataset_seed}.csv")
    original_data_path = pathlib.Path(f"/data/{args.dataset}/{args.data_name}/{args.training_data_name}/training_data.csv")
    print("loading data from", original_data_path)
    data = load(original_data_path)

    with open(args.dataset_config_path, "r") as f:
        configs = json.load(f)["latlon"][dataset_name]
    lat_range = configs["lat_range"]
    lon_range = configs["lon_range"]

    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    grid = Grid(ranges)
    trajs = [[grid.state_to_center_latlon(state) for state in traj] for traj in data]

    # raw_trajs = check_in_range(data, grid)
    # trajs = [[point[1:] for point in traj] for traj in raw_trajs]

    # write data to the file
    save_path = pathlib.Path(f"/data/{args.dataset}/{args.data_name}/{args.training_data_name}//privtrace_training_data.dat")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("writings to", save_path)

    with open(save_path, "w") as f:
        for i in range(len(trajs)):
            if trajs[i] == []:
                continue
            f.write("#"+str(i)+":\n")
            f.write(">0:")
            for j in range(len(trajs[i])):
                f.write(str(trajs[i][j][0])+","+str(trajs[i][j][1])+";")
            f.write("\n")