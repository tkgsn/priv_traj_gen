import osmnx as ox
import os
import sys
import geopandas as gpd

# sys.path.append("../../priv_traj_gen")
from my_utils import load, load_latlon_range
from grid import Grid
from data_pre_processing import check_in_range

def stringify_nonnumeric_cols(gdf):
    import pandas as pd
    for col in (c for c in gdf.columns if not c == "geometry"):
        if not pd.api.types.is_numeric_dtype(gdf[col]):
            gdf[col] = gdf[col].fillna("").astype(str)
    return gdf

    
def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    if 'utils_graph' in dir(ox):        
        gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    elif 'utils' in dir(ox):
        gdf_nodes, gdf_edges = ox.utils.graph_to_gdfs(G)
    else:
        print("Error, graph to gdf function not found")
    gdf_nodes = stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = stringify_nonnumeric_cols(gdf_edges)
    # We need an unique ID for each edge
    # gdf_edges["fid"] = gdf_edges.index
    # Shun: tuple does not work due to the osmnx version?
    gdf_edges["fid"] = [i+1 for i, _ in enumerate(gdf_edges.index)]
    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    # print(gdf_edges)
    gdf_edges.to_file(filepath_edges, encoding=encoding)
    return gdf_edges



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


def make_mtnet_training_data(data_dir, save_dir):

    gdf_edges = gpd.read_file(os.path.join(data_dir, "edges.shp"))

    make_edge_property_file(gdf_edges, save_dir)
    make_edge_adj_file(gdf_edges, save_dir)


def run(dataset, data_path, save_dir):
    lat_range, lon_range = load_latlon_range(dataset)
    print(lat_range, lon_range)

    upper_left = (lat_range[1], lon_range[0])
    upper_right = (lat_range[1], lon_range[1])
    lower_left = (lat_range[0], lon_range[0])
    lower_right = (lat_range[0], lon_range[1])
    print("downloading graph of", upper_left, lower_right)
    G = ox.graph_from_bbox(north=upper_left[0], south=lower_left[0], east=upper_right[1], west=upper_left[1], network_type="drive")
    # save graph
    print("number of nodes", len(G.nodes), "number of edges", len(G.edges))
    print("save graph")
    ox.save_graphml(G, os.path.join(save_dir, "graph.graphml"))
    
    print("save shape files")
    save_graph_shapefile_directional(G, filepath=save_dir)

    # make gps file
    # get(data_path)
    data = load(data_path)
    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, 2)
    grid = Grid(ranges)
    data = check_in_range(data, grid)
    print("save file for map matching in", os.path.join(save_dir, "trips.csv"), "and", os.path.join(save_dir, "times.csv"))
    print(save_dir)
    with open(os.path.join(save_dir, "trips.csv"), "w") as f:
        f.write("id;geom\n")
        for i, traj in enumerate(data):
            wkt = f"{i+1};LINESTRING(" + ",".join([f"{point[2]} {point[1]}" for point in traj]) + ")\n"
            f.write(wkt)
    
    with open(os.path.join(save_dir, "times.csv"), "w") as f:
        f.write("id;time\n")
        for traj in data:
            f.write(",".join([str(point[0]) for point in traj]) + "\n")

    # make directory
    make_mtnet_training_data(save_dir, save_dir)
    # send(save_dir, parent=True)


if __name__ == "__main__":
    dataset = sys.argv[1]
    data_path = sys.argv[2]
    save_dir = sys.argv[3]

    # make save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run(dataset, data_path, save_dir)