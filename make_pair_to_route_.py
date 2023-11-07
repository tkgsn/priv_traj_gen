import osmnx as ox
import networkx as nx
import random
import tqdm
from grid import Grid
import json
import sys
import os
import sqlite3
import struct
sys.path.append('../MTNet_Code/MTNet')

from make_training_data import compensate_edge

def prepare_graph(upper_left, lower_left, upper_right, simplify=True, mode="drive"):

    print("download road networks")
    # get the street network for the area
    # G = ox.graph_from_bbox(north=lat_range[1], south=lat_range[0], east=lon_range[1], west=lon_range[0], network_type='drive')
    G = ox.graph_from_bbox(north=upper_left[0], south=lower_left[0], east=upper_right[1], west=upper_left[1], network_type=mode, simplify=simplify)
    # make the graph undirect
    G = G.to_undirected()
    print("number of nodes: ", len(G.nodes))

    if len(G) > 2**16:
        raise ValueError("the number of nodes is too large")

    # change the name of the nodes to integers from 0 to len(G.nodes)-1
    mapping = {}
    for i, node in enumerate(G.nodes):
        mapping[node] = i
    G = nx.relabel_nodes(G, mapping)

    return G



def find_all_pair_of_paths(G, save_path):
    # find all pairs shortest paths
    paths = nx.all_pairs_dijkstra_path(G, weight='length')

    # create a SQLite3 database to store the paths
    conn = sqlite3.connect(os.path.join(save_path, 'paths.db'))
    c = conn.cursor()

    # create table to store paths
    c.execute('''CREATE TABLE IF NOT EXISTS paths (start_node INTEGER, end_node INTEGER, path BLOB, PRIMARY KEY (start_node, end_node))''')

    # write each path to the database
    for start_node, v in tqdm.tqdm(paths):
        for end_node, path in v.items():
            # convert path to bytes using strcut
            # if len(path) > 128:
                # raise ValueError("the length of the path is too long")
                # SQLite has no limit on the length of the bytes
            path = b''.join(struct.pack('>H', i) for i in path)
            c.execute("INSERT INTO paths VALUES (?, ?, ?)", (start_node, end_node, path))

    # commit changes and close connection
    conn.commit()
    conn.close()


def state_to_graph_nodes(state, c):
    nodes = []

    # if there is no node in the state, then go to the previous state
    while len(nodes) == 0:
        # fetch nodes
        c.execute("SELECT node FROM node_to_state WHERE state=?", (state,))
        nodes = c.fetchall()
        nodes = [node[0] for node in nodes]
        if len(nodes) == 0:
            print("there is no node in the state: ", state)
            print("move to the previous state", state-1)
            if state == 0:
                raise ValueError("there must be a node in the state 0")
        state -= 1

    return nodes

def make_node_to_state(G, grid, save_path):

    conn = sqlite3.connect(os.path.join(save_path, 'paths.db'))
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS node_to_state (node integer, state integer)''')
    for node in tqdm.tqdm(G.nodes):
        lat = G.nodes[node]['y']
        lon = G.nodes[node]['x']
        state = grid.latlon_to_state(lat, lon)
        c.execute("INSERT INTO node_to_state VALUES (?, ?)", (node, state))

    conn.commit()
    conn.close()


def get_route(G, c, start_state, end_state):
    # randomly choosing a node in the state
    start_node = random.choice(state_to_graph_nodes(start_state, c))
    end_node = random.choice(state_to_graph_nodes(end_state, c))

    # retrieve the path from the database
    c.execute("SELECT path FROM paths WHERE start_node=? AND end_node=?", (start_node, end_node))
    
    route = c.fetchone()[0]
    format_string = '>' + 'H' * (len(route) // 2)
    route = struct.unpack(format_string, route)

    # extract the latitude and longitude coordinates of the nodes in the route
    lats = []
    lons = []
    for node in route:
        lat = G.nodes[node]['y']
        lon = G.nodes[node]['x']
        lats.append(lat)
        lons.append(lon)
    
    # return sequence of (lat,lon)
    return list(zip(lats, lons))

def latlon_route_to_state_route(grid, latlon_route, target=None):
    state_route = []
    flag = False
    for latlon in latlon_route:
        state = grid.latlon_to_state(latlon[0], latlon[1])
        if len(state_route) == 0:
            state_route.append(state)
        elif state != state_route[-1]:
            edges = compensate_edge((state_route[-1],state), grid.n_bins)
            states = [edge[1] for edge in edges]
            for state in states:
                state_route.append(state)
                if state == target:
                    flag = True
                    break
            if flag:
                break
    return state_route


if __name__ == "__main__":

    dataset = "geolife"
    n_bins = 30

    lat_range = [39.85, 40.1]
    lon_range = [116.25, 116.5]
    n_bins = 30
    mode = "drive"

    # lat_range = [39.85, 39.86]
    # lon_range = [116.25, 116.26]
    # n_bins = 2

    save_path = f"/data/geolife/pair_to_route/{n_bins}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("make grid")
    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    grid = Grid(ranges)

    upper_left = grid.state_to_center_latlon(0)
    lower_left = grid.state_to_center_latlon(n_bins+2-1)
    upper_right = grid.state_to_center_latlon((n_bins+2)**2-n_bins-2)
    lower_right = grid.state_to_center_latlon((n_bins+2)**2-1)

    print("prepare graph")
    G = prepare_graph(upper_left, lower_left, upper_right, mode=mode)
    print("compute all paths, maybe taking a lot of time")
    find_all_pair_of_paths(G, save_path)

    print("make node_to_state table")
    make_node_to_state(G, grid, save_path)

    # all the pairs of states
    states = list(grid.grids)
    # we don't differentiate the start and end points
    pairs = []
    for i in states:
        for j in states[i+1:]:
            pairs.append((i,j))

    
    # make pair_to_route table
    print(f"find routes of {len(pairs)} pairs of states")
    conn = sqlite3.connect(save_path + f'/paths.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS state_edge_to_route (start_state integer, end_state integer, route text, PRIMARY KEY (start_state, end_state))''')
    for pair in tqdm.tqdm(pairs):
        route = get_route(G, c, pair[0], pair[1])
        state_route = latlon_route_to_state_route(grid, route, target=pair[1])
        c.execute("INSERT INTO state_edge_to_route VALUES (?, ?, ?)", (pair[0], pair[1], str(state_route)))

    conn.commit()
    conn.close()
    print("database for pair_to_route is saved to ", save_path + f"/paths.db")
