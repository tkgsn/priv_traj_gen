import argparse
from logging import getLogger, config
import pickle
from my_utils import get_datadir, save
import json
import pandas as pd
import numpy as np
from grid import Grid
import tqdm
import pathlib
import osmnx as ox
import networkx as nx
# add ../privtrace to python path
import sys
sys.path.append("../privtrace")

# from tools.data_reader import DataReader

# construct a graph using the adjacency information (adjss) with the weights (road_lengths)
# adjss: list of adjacency lists
# i.e., adjss[i] is a list of nodes that are adjacent to node i
# road_lengths: list of road lengths

def construct_graph(adjss, road_lengths):
    G = nx.Graph()
    for i in range(len(adjss)):
        for j in range(len(adjss[i])):
            G.add_edge(i, adjss[i][j], weight=road_lengths[i])
    return G

def post_process_chengdu(trajs):
    df = pd.read_csv("/data/chengdu/raw/edge_adj.txt", header=None).values[:,1:].astype(int)
    # remove -1
    adjss = [[v-1 for v in adjs if v != -1] for adjs in df]

    df = pd.read_csv("/data/chengdu/raw/edge_property.txt", header=None)
    road_lengths = [float(v[1]) for v in df.values]

    G = construct_graph(adjss, road_lengths)

    post_processed = []
    for traj in tqdm.tqdm(trajs):
        source = traj[0]
        target = traj[1]

        shortest_path = nx.shortest_path(G, source=source, target=target, weight='length')
        post_processed.append(shortest_path)
    
    return post_processed


def post_process(dataset, data_name, training_data_name, save_name):

    save_path = get_datadir() / "results" / dataset / data_name / training_data_name / save_name
    with open(save_path / "params.json", "r") as f:
        params = json.load(f)


    if dataset == "chengdu":
        # load generated data

        import pandas as pd
        data_path = save_path / "gene.csv"
        trajs = pd.read_csv(data_path, header=None).values[:, :2].astype(int)
        post_processed = post_process_chengdu(trajs)

    return post_processed


    # if is_real:
    #     if params["algorithm"] == "privtrace":
    #         post_processed_trajs = privtrace_training_post_process(params, n_bins, logger=logger)
    #     elif params["algorithm"] == "meta":
    #         post_processed_trajs = meta_training_post_process(params, n_bins, logger=logger)
        
    # else:
        
    #     if params["algorithm"] == "privtrace":
    #         post_processed_trajs = privtrace_post_process(params, n_bins, logger=logger)
    #     elif params["algorithm"] == "meta":
    #         post_processed_trajs = meta_post_process(params, n_bins, logger=logger)

    return post_processed_trajs
    

def meta_post_process(params, n_bins, *, logger):
   # load the dataset generated by meta
    gene_traj = load_dataset(pathlib.Path(get_datadir()) / f"results" / params["dataset"] / params["data_name"] / params["training_data_name"] / params["save_name"] / f"gene.csv", logger=logger)

    logger.info("load graph")
    # load graph of the city in the range of lat_range and lon_range
    lat_range = params["lat_range"]
    lon_range = params["lon_range"]
    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, params['n_bins'])
    original_grid = Grid(ranges)
    
    if n_bins == params['n_bins']:
        logger.info("used grid is the same as the original grid")
        post_process_grid = original_grid
    else:
        logger.info(f"make grid with {params['lat_range']}, {params['lon_range']}, {n_bins}")
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        post_process_grid = Grid(ranges)

    G = ox.graph_from_bbox(lat_range[1], lat_range[0], lon_range[0], lon_range[1], network_type='drive')

    logger.info("complement trajectories")
    our_trajs = []
    for i in tqdm.tqdm(range(len(gene_traj))):
        if len(gene_traj[i]) == 1:
            continue
        location_1 = original_grid.state_to_center_latlon(gene_traj[i][0])
        location_2 = original_grid.state_to_center_latlon(gene_traj[i][1])

        # compute the Euclidean distance between location_1 and location_2
        dist = ox.distance.euclidean_dist_vec(location_1[0], location_1[1], location_2[0], location_2[1])

        # find the nearest node to location_1
        node_1 = ox.nearest_nodes(G, location_1[1], location_1[0])
        # find the nearest node to location_2
        node_2 = ox.nearest_nodes(G, location_2[1], location_2[0])
        # find the shortest path
        try:
            path = nx.shortest_path(G, node_1, node_2, weight='length')
        except:
            # print("a")
            continue

        traj = []
        for node in path:
            traj.append(post_process_grid.latlon_to_state(G.nodes[node]['y'], G.nodes[node]['x']))

        our_trajs.append([traj[0]] + [traj[i] for i in range(1, len(traj)) if traj[i] != traj[i-1]])
    
    return our_trajs

def privtrace_state_to_latlon(state, f):
    lat_left, lat_right, lon_left, lon_right = f[0][state]
    # sample a lat and lon from the range
    lat = np.random.uniform(lat_left, lat_right)
    lon = np.random.uniform(lon_left, lon_right)

    # use the center of the range
    # lat = (lat_left + lat_right) / 2
    # lon = (lon_left + lon_right) / 2
    return lat, lon

def privtrace_training_post_process(params, n_bins, *, logger):
    # load dataset used by training privtrace
    reader1 = DataReader()

    lat_range = params["lat_range"]
    lon_range = params["lon_range"]
    logger.info(f"make grid with {params['lat_range']}, {params['lon_range']}, {n_bins}")
    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    grid = Grid(ranges)

    load_path = get_datadir() / "results" / params["dataset"] / params["data_name"] / params["training_data_name"] / "training_data.csv"
    tr_list = reader1.read_trajectories_from_data_file(load_path)
    # convert tr_list to state list
    trajs = []
    for tr in tqdm.tqdm(tr_list):
        traj = [grid.latlon_to_state(lat, lon) for lon, lat in tr]
        trajs.append(traj)

    real_state_trajs = []
    for traj in trajs:
        real_state_trajs.append([traj[0]] + [traj[i] for i in range(1, len(traj)) if traj[i] != traj[i-1]])

    return real_state_trajs

def meta_training_post_process(params, n_bins, *, logger):
    # load the training dataset
    if n_bins == params['n_bins']:
        logger.info("return the original dataset because the number of bins is the same as the original one")
        gene_trajs = load_dataset(pathlib.Path(get_datadir()) / params["dataset"] / params["data_name"] / params["training_data_name"] / f"training_data.csv", logger=logger)
    else:
        print("do something")
    
    return gene_trajs


def privtrace_post_process(params, n_bins, *, logger):
    privtrace_save_dir = get_datadir() / "results" / params["dataset"] / params["data_name"] / params["training_data_name"] / params["save_name"]
    logger.info(f"load files from {privtrace_save_dir}")
    with open(privtrace_save_dir / f"grid_info.pickle", "rb") as f:
        f = pickle.load(f)
    df = pd.read_csv(privtrace_save_dir / f"gene.csv", header=None)
    trajs = df.values.tolist()

    # remove the value 10000
    trajs = [[x for x in traj if x != 10000] for traj in trajs]
    privtrace_gene_trajs = [[privtrace_state_to_latlon(state, f) for state in traj] for traj in trajs]

    lat_range = params["lat_range"]
    lon_range = params["lon_range"]
    logger.info(f"make grid with {params['lat_range']}, {params['lon_range']}, {n_bins}")
    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    grid = Grid(ranges)

    privtrace_state_trajs = []
    # convert tr_list to state list
    for traj in tqdm.tqdm(privtrace_gene_trajs):
        traj = [grid.latlon_to_state(lat, lon) for lat, lon in traj]
        privtrace_state_trajs.append(traj)
    
    for j, traj in enumerate(privtrace_state_trajs):
        privtrace_state_trajs[j] = [traj[0]] + [traj[i] for i in range(1, len(traj)) if traj[i] != traj[i-1]]

    return privtrace_state_trajs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--training_data_name', type=str)
    parser.add_argument('--is_real', action='store_true')
    args = parser.parse_args()
    
    save_dir = get_datadir() / "results" / args.dataset / args.data_name / args.training_data_name / args.save_name
    # set logger
    with open('./log_config.json', 'r') as f:
        log_conf = json.load(f)
    log_conf["handlers"]["fileHandler"]["filename"] = str(save_dir / "post_process.log")
    config.dictConfig(log_conf)
    logger = getLogger(__name__)
    logger.info('log is saved to {}'.format(save_dir / "post_process.log"))
    logger.info(f'used parameters {vars(args)}')


    if args.is_real:
        save_path = save_dir.parent / 'post_processed_trajs_real.csv'
    else:
        save_path = save_dir / 'post_processed_trajs.csv'
    
    if save_path.exists():
        logger.info(f"{save_path} already exists")
        sys.exit()
    else:
        post_processed_trajs = post_process(args.dataset, args.data_name, args.training_data_name, args.save_name, 0, args.is_real, logger=logger)
        logger.info(f"save post processed trajs to {save_path}")
        save(save_path, post_processed_trajs)