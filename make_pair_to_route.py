import networkx as nx
import sqlite3
import pathlib
import tqdm
import json
from grid import Grid
import shapely.wkt
import concurrent.futures
import functools
import pickle

def load_edges(data_dir):
    """
    return a list of edges
    edge is a pair of latlon and distance
    """
    property_path = pathlib.Path(data_dir) / "edge_property.txt"
    nodes_edges = [[]]
    with open(property_path, "r") as f:
        for i, line in enumerate(f):
            distance = float(line.split(",")[1])
            wkt = ",".join(line.split(",")[4:])[1:-1]
            # load wkt by sharpley
            wkt = shapely.wkt.loads(wkt)
            lonlats = wkt.coords.xy
            lonlats = list(zip(lonlats[0], lonlats[1]))
            # lonlats = line.split("LINESTRING")[1][1:-3].split(",")
            # lonlats = [list(map(float,lonlat.split())) for lonlat in lonlats]
            start_lonlat = tuple(lonlats[0])
            end_lonlat = tuple(lonlats[-1])

            nodes_edges.append([start_lonlat[::-1], end_lonlat[::-1], distance])
    
    return nodes_edges


def make_graph(data_dir):
    """
    make a undirected graph and a directed graph from the Chengdu dataset
    """

    # make a undirected graph
    # G = nx.Graph()
    # make a directed graph
    DG = nx.DiGraph()

    # add nodes
    nodes_edges = load_edges(data_dir)
    for start_latlon, end_latlon, distance in nodes_edges[1:]:
        # G.add_node(start_latlon)
        # G.add_node(end_latlon)
        DG.add_node(start_latlon)
        DG.add_node(end_latlon)
    
    # add edges
    edge_path = pathlib.Path(data_dir) / "edge_adj.txt"
    with open(edge_path, "r") as f:
        for i, (line, edge) in enumerate(zip(f, nodes_edges[1:])):
            adjs = line.split(",")[1:-1]
            adjs = [v for v in list(map(int, adjs)) if v != -1]

            for adj in adjs:
                assert edge[1] == nodes_edges[adj][0], f"{edge[1]} != {nodes_edges[adj][0]}"

            # G.add_edge(edge[0], edge[1], weight=edge[2])
            # G.add_edge(edge[1], edge[0], weight=edge[2])
            DG.add_edge(edge[0], edge[1], weight=edge[2])

    # return DG, G
    return DG


# def find_the_node_from_latlon(G, latlon):
#     """
#     find the nearest node from latlon
#     """
#     nodes = list(G.nodes)
#     min_distance = float("inf")
#     min_node = None
#     for node in nodes:
#         distance = geodesic(latlon, node).km
#         if distance < min_distance:
#             min_distance = distance
#             min_node = node
#     return min_node

# def make_state_to_node(G, n_states, state_to_latlon, db_path):
#     """
#     make a mapping from state to node
#     mapping is based on the shortest euclidean distance
#     """

#     with sqlite3.connect(db_path) as conn:
#         c = conn.cursor()
#         c.execute("DROP TABLE IF EXISTS state_to_node")
#         c.execute("CREATE TABLE IF NOT EXISTS state_to_node (state integer, node text, PRIMARY KEY (state))")

#         for i in tqdm.tqdm(range(n_states)):
#             latlon = state_to_latlon(i)
#             node = find_the_node_from_latlon(G, latlon)
#             c.execute("INSERT INTO state_to_node VALUES (?, ?)", (i, str(node)))


def make_node_to_state(G, n_states, latlon_to_state, db_path):
    """
    make a mapping from node to state
    mapping is based on the shortest euclidean distance
    """

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS node_to_state")
        c.execute("CREATE TABLE IF NOT EXISTS node_to_state (node text, state integer, PRIMARY KEY (node))")

        for node in G:
            state = latlon_to_state(*node)
            c.execute("INSERT INTO node_to_state VALUES (?, ?)", (str(node), state))


def state_pair_to_latlon_routes(state_pair, cursor):

    # c = cursor.execute("SELECT * FROM state_to_node WHERE state=?", (state_pair[0],))
    # start_node = c.fetchone()[1]
    # c = cursor.execute("SELECT * FROM state_to_node WHERE state=?", (state_pair[1],))
    # end_node = c.fetchone()[1]
    c = cursor.execute("SELECT node FROM node_to_state WHERE state=?", (state_pair[0],))
    # start_node = c.fetchone()[1]
    start_nodes = c.fetchall()
    c = cursor.execute("SELECT node FROM node_to_state WHERE state=?", (state_pair[1],))
    # end_nodes = c.fetchone()[1]
    end_nodes = c.fetchall()
    
    latlon_routes = []
    for start_node in start_nodes:
        start_node = eval(start_node[0])
        for end_node in end_nodes:
            end_node = eval(end_node[0])
            c = cursor.execute("SELECT * FROM paths WHERE start_node=? AND end_node=?", (str(start_node), str(end_node)))
            latlon_route = c.fetchone()
            if latlon_route is not None:
                latlon_route = eval(latlon_route[2])
                latlon_routes.append(latlon_route)
    
    # choose the shortest one
    # if len(latlon_routes) == 0:
        # return None
    # latlon_route = min(latlon_routes, key=lambda x: len(x))

    # c = cursor.execute("SELECT * FROM paths WHERE start_node=? AND end_node=?", (str(start_node), str(end_node)))
    # latlon_route = c.fetchone()
    # if latlon_route is not None:
        # latlon_route = eval(latlon_route[2])
    
    return latlon_routes

def latlon_route_to_state_route(latlon_route, latlon_to_state):
    state_route = [latlon_to_state(latlon_route[0][0], latlon_route[0][1])]
    for latlon in latlon_route[1:]:
        state = latlon_to_state(latlon[0], latlon[1])
        if state != state_route[-1]:
            state_route.append(state)
    return state_route


# def make_paths(DG, db_path):
#     """
#     make a mapping from node to node
#     mapping is based on the shortest euclidean distance
#     """
#     # paths = nx.all_pairs_dijkstra_path(G)
#     di_paths = nx.all_pairs_dijkstra_path(DG)

#     # print("find paths for all pairs from", len(G), "nodes")
#     print("find paths for all pairs from", len(DG), "nodes")
#     with sqlite3.connect(db_path) as conn:
#         c = conn.cursor()
#         c.execute("DROP TABLE IF EXISTS paths")
#         c.execute("CREATE TABLE paths (start_node text, end_node text, path text, PRIMARY KEY (start_node, end_node))")

#         # for start_node, v in tqdm.tqdm(paths):
#             # for end_node, path in v.items():
#                 # c.execute("INSERT INTO paths VALUES (?, ?, ?)", (str(start_node), str(end_node), str(path)))
        
#         for start_node, v in tqdm.tqdm(di_paths):
#             for end_node, path in v.items():
#                 # if start_node, end_node in paths, then skip
#                 # c.execute("SELECT * FROM paths WHERE start_node=? AND end_node=?", (str(start_node), str(end_node)))
#                 # if c.fetchone() != None:
#                     # continue
#                 c.execute("INSERT INTO paths VALUES (?, ?, ?)", (str(start_node), str(end_node), str(path)))



def check_node_in_state(cursor, state):
    c = cursor.execute("SELECT * FROM node_to_state WHERE state=?", (state,))
    node = c.fetchall()
    if len(node) == 0:
        return None
    else:
        return [eval(n[0]) for n in node]


# def process_state_i_(i, states, db_path, latlon_to_state, DG):
#     n_inserted = 0
#     with sqlite3.connect(db_path) as conn:
#         c = conn.cursor()

#         nodes = check_node_in_state(c, i)
#         # compute path from node
#         paths = []
#         for node in nodes:
#             paths.append(nx.single_source_dijkstra_path(DG, node))

#         for j in states:
#             latlon_routes = []

#             if i == j:
#                 continue

#             end_nodes = check_node_in_state(c, j)
#             if end_nodes is None:
#                 # print("WARNING", j, "has no node")
#                 continue

#             for path in paths:
#                 for end_node in end_nodes:
#                     if end_node in path:
#                         latlon_routes.append(path[end_node])


#             # print(latlon_routes)
#             state_routes = []
#             for latlon_route in latlon_routes:
#                 state_route = latlon_route_to_state_route(latlon_route, latlon_to_state)
#                 assert state_route[0] == i, f"different start point {i} {j} -> {state_route}"
#                 assert state_route[-1] == j, f"different end point {i} {j} -> {state_route}"
#                 state_routes.append(state_route)

#             # remove duplicate routes
#             state_routes = list(set([tuple(route) for route in state_routes]))
#             n_inserted += 1
#             c.execute("INSERT INTO state_edge_to_route VALUES (?, ?, ?)", (i, j, str(state_routes)))
#     return n_inserted

def process_state_i(i, states, db_path, latlon_to_state, DG, truncate):
    """
    find the shortest path from state i to all other states
    """

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()

        nodes = check_node_in_state(c, i)
        # compute path from node
        paths = []
        for node in nodes:
            # paths.append(nx.single_source_dijkstra_path(DG, node))
            paths.append(nx.single_source_dijkstra(DG, node))

        for j in states:

            if i == j:
                continue


            end_nodes = check_node_in_state(c, j)
            if end_nodes is None:
                # print("WARNING", j, "has no node")
                continue

            shortest_path = None
            shortest_length = float("inf")
            flag = False
            # find the shortest path to end_nodes
            for (length, path), node in zip(paths, nodes):
                for end_node in end_nodes:
                    # check if (node, end_node) is in DG
                    if (node, end_node) in DG.edges:
                        flag = True
                        break
                    if end_node in path:
                        if length[end_node] < shortest_length:
                            shortest_length = length[end_node]
                            shortest_path = path[end_node]
                if flag:
                    # in this case, there is a direct road from node to end_node
                    break

            # for path in paths:
            #     for end_node in end_nodes:
            #         if end_node in path:
            #             latlon_routes.append(path[end_node])


            # print(latlon_routes)
            # state_routes = []
            # for latlon_route in latlon_routes:
            # if len(shortest_path) > truncate:
                # print(shortest_path)
            if (shortest_path is not None) and (len(shortest_path) < truncate):
                if flag:
                    state_route = [i, j]
                    assert len(state_route) == 2, f"direct road but not length 2 {i} {j} -> {state_route}"
                else:
                    state_route = latlon_route_to_state_route(shortest_path, latlon_to_state)
                    assert state_route[0] == i, f"different start point {i} {j} -> {state_route}"
                    assert state_route[-1] == j, f"different end point {i} {j} -> {state_route}"
                # state_routes.append(state_route)

                # remove duplicate routes
                # state_routes = list(set([tuple(route) for route in state_routes]))
                # state_routess[j] = state_routes

                # save with pickle
                with open(f"temp/state_routes_from_{i}_to_{j}.pkl", "wb") as f:
                    pickle.dump(state_route, f)

                # c.execute("INSERT INTO state_edge_to_route VALUES (?, ?, ?)", (i, j, str(state_routes)))

def make_state_pair_to_state_route(n_states, db_path, latlon_to_state, DG, truncate):

    states = list(range(n_states))
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()

        # find the possible states as start state
        start_states = []
        for i in states:
            nodes = check_node_in_state(c, i)
            if nodes is None:
                print("WARNING", i, "has no node")
                continue
            start_states.append(i)

    partial_process_state_i = functools.partial(process_state_i, states=states, db_path=db_path, latlon_to_state=latlon_to_state, DG=DG, truncate=truncate)
    # for i in tqdm.tqdm(start_states):
        # partial_process_state_i(i)
    
    pathlib.Path("temp").mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # iterator = executor.map(partial_process_state_i, start_states)
        futures = [executor.submit(partial_process_state_i, i) for i in start_states]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()
    
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS state_edge_to_route")
        c.execute("CREATE TABLE IF NOT EXISTS state_edge_to_route (start_state integer, end_state integer, route text, PRIMARY KEY (start_state, end_state))")
        # write to db
        for i in tqdm.tqdm(start_states):
            for j in states:
                if pathlib.Path(f"temp/state_routes_from_{i}_to_{j}.pkl").exists():
                    with open(f"temp/state_routes_from_{i}_to_{j}.pkl", "rb") as f:
                        state_routes = pickle.load(f)
                    c.execute("INSERT INTO state_edge_to_route VALUES (?, ?, ?)", (i, j, str(state_routes)))



def run(n_bins, data_dir, lat_range, lon_range, truncate, save_dir):
    if truncate == 0:
        truncate = float("inf")

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    db_path = pathlib.Path(save_dir) / "paths.db"

    print("make grid")
    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    grid = Grid(ranges)

    n_states = len(grid.grids)

    print("make graph")
    DG = make_graph(data_dir)

    print("make node_to_state")
    make_node_to_state(DG, n_states, grid.latlon_to_state, db_path)

    print("make state_pair_to_state_route to", db_path)
    make_state_pair_to_state_route(n_states, db_path, grid.latlon_to_state, DG, truncate)