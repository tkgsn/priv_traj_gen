import json
import pathlib
import sys
import os
import shapely.wkt

sys.path.append("../../priv_traj_gen")
from grid import Grid
# generated data format:
# 3677,3681,3837,3681,3838,3844,4001,4005,0,0,0,0,0,0,0,0,0,0,0,0,0

def make_edge_to_state_pair(data_path, latlon_config_path, n_bins):
    # make id_to_edge.json
    with open(latlon_config_path, "r") as f:
        param = json.load(f)
    lat_range = param["lat_range"]
    lon_range = param["lon_range"]

    print("make grid of ", lat_range, lon_range, n_bins)
    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    grid = Grid(ranges)
    # edge is a tuple of states (first state, last state)
    # 1,0,0,0,LINESTRING"(39.72916666666667 116.14250000000001,39.72916666666667 116.14250000000001)"
    # first latlon (39.72916666666667 116.14250000000001) -> first state
    # last latlon (39.72916666666667 116.14250000000001) -> last state
    edge_id_to_state_pair = {}
    print(os.path.join(data_path, "edge_property.txt"))
    with open(os.path.join(data_path, "edge_property.txt"), "r") as f:
        for i, line in enumerate(f):
            # load wkt by shapely
            wkt = ",".join(line.split(",")[4:])
            wkt = shapely.wkt.loads(wkt)
            # get the first lon,lat
            from_lonlat = wkt.coords[0]
            # get the last lon,lat
            to_lonlat = wkt.coords[-1]
            from_state = grid.latlon_to_state(*from_lonlat[::-1])
            if from_state == None:
                print(*from_lonlat[::-1], line)
                raise
            to_state = grid.latlon_to_state(*to_lonlat[::-1])
            if to_state == None:
                print("w", line)
                raise
            edge = (from_state, to_state)
            edge_id_to_state_pair[i+1] = edge
    
    return edge_id_to_state_pair, grid

def convert_to_original_format(traj_path, time_path, edge_id_to_state_pair):

    # load data
    trajs = []
    with open(pathlib.Path(traj_path), "r") as f:
        for line in f:
            traj = [int(vocab) for vocab in line.split(",")]
            # traj = [int(vocab) for vocab in line.strip().split(" ")]
            # remove 0s at the end
            traj = [vocab for vocab in traj if vocab != 0]
            trajs.append(traj)

    new_trajs = []
    for traj in trajs:
        new_traj = []
        for i in range(len(traj)):
            if i == 0:
                new_traj.append(edge_id_to_state_pair[traj[i]][0])
                new_traj.append(edge_id_to_state_pair[traj[i]][1])
            else:
                new_traj.append(edge_id_to_state_pair[traj[i]][1])
        new_trajs.append(new_traj)

    new_time_trajs = []

    return new_trajs, new_time_trajs


if __name__ == "__main__":

    # find generated data named samples_*.txt from directory given by the argument
    training_data_dir = pathlib.Path(sys.argv[1])
    save_dir = pathlib.Path(sys.argv[2])
    save_dir.mkdir(parents=True, exist_ok=True)
    latlon_config_path = pathlib.Path(sys.argv[3])
    n_bins = int(sys.argv[4])

    edge_to_state_pair, _ = make_edge_to_state_pair(training_data_dir, latlon_config_path, n_bins)

    print(save_dir)
    files = [file for file in save_dir.iterdir() if file.name.startswith("samples_t_")]
    # sort
    files.sort(key=lambda x: int(x.name.split("_")[1].split(".")[0]))
    for file in files:
        # get the id
        id = file.name.split("_")[1].split(".")[0]
        traj_file = file.parent / f"samples_{i}.txt"
        print("convert to original format: ", file, "and", traj_file)
        trajs, new_time_trajs = convert_to_original_format(traj_file, file, edge_to_state_pair)
        print("save to", save_dir / f"generated_{id}.csv")
        with open(save_dir / f"generated_{id}.csv", "w") as f:
            for traj in trajs:
                f.write(",".join([str(vocab) for vocab in traj]) + "\n")