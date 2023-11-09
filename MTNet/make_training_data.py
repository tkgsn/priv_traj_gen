
import sys
import pathlib
import json
import numpy as np
import geopandas as gpd
import os

# sys.path.append('../../priv_traj_gen')
# from my_utils import load
# from grid import Grid

# def run(path, save_path):
#     # load
#     with open(path.parent / "params.json", "r") as f:
#         param = json.load(f)
#     n_bins = param["n_bins"]
#     lat_range = param["lat_range"]
#     lon_range = param["lon_range"]
#     distance_matrix = np.load(path.parent.parent.parent / f"distance_matrix_bin{n_bins}.npy")

#     edges, edges_properties, adjs = make_edge_properties(lat_range, lon_range, n_bins, distance_matrix)
#     # make edge_property file
#     with open(save_path / "edge_property.txt", "w") as f:
#         for i in range(1, len(edges_properties)+1):
#             from_lat = edges_properties[i-1][3][0][0]
#             from_lon = edges_properties[i-1][3][0][1]
#             to_lat = edges_properties[i-1][3][1][0]
#             to_lon = edges_properties[i-1][3][1][1]
#             f.write(f'{i},{edges_properties[i-1][0]},{edges_properties[i-1][1]},{edges_properties[i-1][2]},"LINESTRING({from_lat} {from_lon},{to_lat} {to_lon})"\n')
    
#     # make id_to_edge file (edge is (from_state, to_state))
#     id_to_edge = {}
#     for i in range(1, len(edges_properties)+1):
#         id_to_edge[i] = edges[i-1]
#     with open(save_path / "id_to_edge.json", "w") as f:
#         json.dump(id_to_edge, f)
#     edge_to_id = {v:k for k,v in id_to_edge.items()}

#     # make adjs file
#     max_n_adjs = 4
#     with open(save_path / "edge_adj.txt", "w") as f:
#         for edge in edges:
#             end_location = edge[-1]
#             if len(edge) == 1:
#                 adj_edges = adjs[end_location]
#             else:
#                 adj_edges = adjs[end_location]
#                 # remove the edge that reverse the direction
#                 # adj_edges = [adj_edge for adj_edge in adj_edges if adj_edge != (edge[1],edge[0])]
            
#             adj_edge_ids = [edge_to_id[adj_edge] for adj_edge in adj_edges]
#             # padding with -1
#             adj_edge_ids.extend([-1]*(max_n_adjs-len(adj_edge_ids)))
#             f.write(f',{",".join([str(adj_edge_id) for adj_edge_id in adj_edge_ids])}\n')

#     # make trajectory file
#     trajectories = load(path)
#     trajs = convert(trajectories, edge_to_id, n_bins)
#     with open(save_path / "trajs_demo.csv", "w") as f:
#         for traj in trajs:
#             f.write(" ".join([str(vocab) for vocab in traj] + [str(0)])+"\n")

#     # make time file
#     time_trajectories = load(pathlib.Path(path).parent / "training_data_time.csv")
#     time_trajs = convert_time(time_trajectories)
#     with open(save_path / "tstamps_demo.csv", "w") as f:
#         for traj in time_trajs:
#             f.write(" ".join([str(vocab) for vocab in traj])+"\n")

def convert(trajectories, edge_to_id, n_bins):
    
    new_trajectories = []
    # we first convert a trajectory to a list of edges
    for traj in trajectories:
        # edge_traj has a special vocab in the first place that represents the start place of the trajectory
        new_traj = [edge_to_id[(traj[0],)]]
        # compensate and convert
        edge_traj = convert_traj_to_edges(traj, n_bins)
        for edge in edge_traj:
            if edge_to_id[edge] != new_traj[-1]:
                new_traj.append(edge_to_id[edge])
        
        new_trajectories.append(new_traj)
    
    return new_trajectories

def convert_time(time_trajectories):
    new_trajectories = []
    for traj in time_trajectories:
        new_traj = [0]
        for i in range(len(traj)-1):
            new_traj.append(traj[i+1]-traj[i])
        new_trajectories.append(new_traj)
    return new_trajectories

def convert_traj_to_edges(traj, n_bins):
    new_traj = []
    for i in range(len(traj)-1):
        from_state = traj[i]
        to_state = traj[i+1]
        edge = (from_state,to_state)
        new_edges = compensate_edge(edge, n_bins)

        new_traj.extend(new_edges)
    return new_traj


def compensate_edge(edge, n_bins):
    '''
    In the case, the edge is not neighboring, we compensate the edge by adding the edges between the two states
    the route is the hamming way, from the direction of x-axis to the direction of y-axis
    '''

    from_state = edge[0]
    to_state = edge[1]
    from_x = from_state % (n_bins+2)
    from_y = from_state // (n_bins+2)
    to_x = to_state % (n_bins+2)
    to_y = to_state // (n_bins+2)
    x_sign = 1 if from_x < to_x else -1
    y_sign = 1 if from_y < to_y else -1

    edges = []
    for x in range(from_x, to_x, x_sign):
        edges.append((from_y*(n_bins+2)+x, from_y*(n_bins+2)+x+x_sign))
    
    for y in range(from_y, to_y, y_sign):
        edges.append((y*(n_bins+2)+to_x, (y+y_sign)*(n_bins+2)+to_x))

    return edges


# def make_edge_properties(lat_range, lon_range, n_bins, distance_matrix):

#     print("make grid of ", lat_range, lon_range, n_bins)
#     ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
#     grid = Grid(ranges)
#     edges, adjs = make_edges(n_bins)
#     aux_infos = [add_aux_info_to_edge(edge, distance_matrix, grid.state_to_center_latlon) for edge in edges]

#     return edges, aux_infos, adjs

def make_edges(n_bins):
    # edges are made in the order of state
    edges = []
    adjs = {state:[] for state in range((n_bins+2)**2)}
    for y in range(n_bins+2):
        for x in range(n_bins+2):
            state = y * (n_bins+2) + x
            edges.append((state,))
            if x -1 >= 0:
                edge = (state,state-1)
                if edge not in edges:
                    edges.append(edge)
                adjs[state].append(edge)
            if y -1 >= 0:
                edge = (state,state-n_bins-2)
                if edge not in edges:
                    edges.append(edge)
                adjs[state].append(edge)
            if x <= n_bins:
                edge = (state,state+1)
                if edge not in edges:
                    edges.append(edge)
                adjs[state].append(edge)
            if y <= n_bins:
                edge = (state,state+n_bins+2)
                if edge not in edges:
                    edges.append(edge)
                adjs[state].append(edge)

    return edges, adjs

def add_aux_info_to_edge(edge, distance_matrix, state_to_latlon):
    # All WKTs have two length
    # 2 road types (start, move)
    # 3 types of headding, north, east south, we consider the headding of the start vocab as 0
    # length is the Euclidian distance of the two centers of the grids
    n_locations_in_x = int(np.sqrt(distance_matrix.shape[0]))
    if len(edge) == 1:
        from_latlon = state_to_latlon(edge[0])
        to_latlon = state_to_latlon(edge[0]) 
        heading = 0
        road_type = 0
        length = 0      
    elif len(edge) == 2:
        from_latlon = state_to_latlon(edge[0])
        to_latlon = state_to_latlon(edge[1])
        if edge[0] == edge[1] - 1:
            heading = 90
        elif edge[0] == edge[1] + 1:
            heading = 270
        elif edge[0] == edge[1] + n_locations_in_x:
            heading = 180
        elif edge[0] == edge[1] - n_locations_in_x:
            heading = 0
        else:
            raise ValueError("edge must be neighboring")
        road_type = 1
        length = distance_matrix[edge[0]][edge[1]]
    else:
        raise ValueError("edge length must be 1 or 2")
    return length, road_type, heading, (from_latlon, to_latlon)

def make_edge_property_file(gdf_edges, save_dir):
    # 1,0,0,0,LINESTRING"(39.72916666666667 116.14250000000001,39.72916666666667 116.14250000000001)"
    # id, road_type, heading, length, wkt
    # we ignore heading
    
    # make road type categories
    road_types = set(gdf_edges["highway"])
    road_type_to_id = {road_type:i for i, road_type in enumerate(road_types)}

    with open(os.path.join(save_dir, "edge_property.txt"), "w") as f:
        for i, row in gdf_edges.iterrows():
            wkt = row["geometry"]
            road_type = road_type_to_id[row["highway"]]
            length = row["length"]
            f.write(f'{i+1},{road_type},0,{length},"{wkt}"\n')

def make_edge_adj_file(gdf_edges, save_dir):
    # ,2,3,-1,-1
    # line id-1: adj1, adj2, adj3, adj4, -1, ..., -1
    # -1 means no adj

    adjss = []
    for i, row in gdf_edges.iterrows():
        is_end_node = row["v"]
        adjs = gdf_edges[gdf_edges["u"] == is_end_node]["fid"]
        adjs = [adj for adj in adjs.tolist()]
        adjss.append(adjs)

    max_num_adjs = max([len(adjs) for adjs in adjss])
    with open(os.path.join(save_dir, "edge_adj.txt"), "w") as f:
        for adjs in adjss:
            adjs.extend([-1]*(max_num_adjs-len(adjs)))
            f.write("," + ",".join([str(adj) for adj in adjs])+"\n")

def convert_mr_to_training(data_dir, save_dir):
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

            training_data_time.append(change_times)
            training_data.append(edge_ids + [0])
            assert len(change_times) == len(edge_ids), f"{len(change_times)} != {len(edge_ids)}"
    
    with open(os.path.join(save_dir, "training_data.csv"), "w") as f:
        for edge_ids in training_data:
            f.write(" ".join([str(edge_id) for edge_id in edge_ids])+"\n")
    
    with open(os.path.join(save_dir, "training_data_time.csv"), "w") as f:
        for times in training_data_time:
            f.write(" ".join([str(time) for time in times])+"\n")


def run_geolife(data_dir, save_dir):

    gdf_edges = gpd.read_file(os.path.join(data_dir, "edges.shp"))
    print(gdf_edges)

    make_edge_property_file(gdf_edges, save_dir)
    make_edge_adj_file(gdf_edges, save_dir)
    convert_mr_to_training(data_dir, save_dir)

def run_chengdu(data_path, save_path, num_data, seed):

    # # make id_to_edge.json
    # with open(setting_path, "r") as f:
    #     param = json.load(f)
    # n_bins = param["n_bins"]
    # lat_range = param["lat_range"]
    # lon_range = param["lon_range"]
    # print("make grid of ", lat_range, lon_range, n_bins)
    # ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    # grid = Grid(ranges)
    # # edge is a tuple of states (first state, last state)
    # # 1,0,0,0,LINESTRING"(39.72916666666667 116.14250000000001,39.72916666666667 116.14250000000001)"
    # # first latlon (39.72916666666667 116.14250000000001) -> first state
    # # last latlon (39.72916666666667 116.14250000000001) -> last state
    # id_to_edge = {}
    # with open(data_path / "edge_property.txt", "r") as f:
    #     for i, line in enumerate(f):
    #         wkt = line.split("LINESTRING")[-1]
    #         lonlats = wkt.split(",")
    #         from_lonlat = lonlats[0].split("(")[-1]
    #         to_lonlat = lonlats[-1].split(")")[0]
    #         from_lonlat = tuple([float(vocab) for vocab in from_lonlat.split(" ")])
    #         to_lonlat = tuple([float(vocab) for vocab in to_lonlat.split(" ")])
    #         from_state = grid.latlon_to_state(*from_lonlat[::-1])
    #         if from_state == None:
    #             print(*from_lonlat[::-1], line)
    #             raise
    #         to_state = grid.latlon_to_state(*to_lonlat[::-1])
    #         if to_state == None:
    #             print("w", line)
    #             raise
    #         edge = (from_state, to_state)
    #         id_to_edge[i+1] = edge
    # with open(save_path / "id_to_edge.json", "w") as f:
    #     json.dump(id_to_edge, f)
    # print("save id_to_edge.json to", save_path / "id_to_edge.json")

    # copy data_path / edge_property.txt to save_path / edge_property.txt
    with open(data_path / "edge_property.txt", "r") as f:
        lines = f.readlines()
    with open(save_path / "edge_property.txt", "w") as f:
        for line in lines:
            f.write(line)

    # copy data_path / edge_adj.txt to save_path / edge_adj.txt
    with open(data_path / "edge_adj.txt", "r") as f:
        lines = f.readlines()
    with open(save_path / "edge_adj.txt", "w") as f:
        for line in lines:
            f.write(line)

    # load data_path / trajs_demo.csv
    with open(data_path / "trajs_demo.csv", "r") as f:
        lines = f.readlines()

    # load data_path / tstamps_demo.csv
    with open(data_path / "tstamps_demo.csv", "r") as f:
        time_lines = f.readlines()

    # shuffle
    np.random.seed(seed)
    if num_data != 0:
        print("shuffle trajectories and choose the first", num_data, "trajectories")
        # shuffle trajectories and real_time_traj with the same order without using numpy
        p = np.random.permutation(len(lines))
        lines = [lines[i] for i in p]
        lines = lines[:num_data]
        time_lines = [time_lines[i] for i in p]
        time_lines = time_lines[:num_data]

    
    # write
    with open(save_path / "training_data.csv", "w") as f:
        for line in lines:
            f.write(line)

    with open(save_path / "training_data_time.csv", "w") as f:
        for line in time_lines:
            f.write(line)

    

if __name__ == "__main__":
    data_path = pathlib.Path(sys.argv[1])
    save_path = pathlib.Path(sys.argv[2])
    save_path.mkdir(parents=True, exist_ok=True)

    dataset = sys.argv[3]
    
    if dataset == "chengdu":

        print(data_path, save_path, dataset)
        num_data = int(sys.argv[4])
        seed = int(sys.argv[5])
        # setting_path = pathlib.Path(sys.argv[6])

        # run_chengdu(data_path, save_path, num_data, seed, setting_path)
        run_chengdu(data_path, save_path, num_data, seed)
    elif dataset == "geolife":
        run_geolife(data_path, save_path)
    elif dataset == "geolife_test":
        run_geolife(data_path, save_path)
    # else:
    #     run(data_path, save_path)