import unittest
import sys
import folium
import sqlite3
import json
import networkx as nx
import os

sys.path.append('./')
import make_pair_to_route
from grid import Grid

class TestDatabase(unittest.TestCase):
    def setUp(self):
        n_bins = 30
        dataset = "chengdu"
        self.route_db_path = f"/data/{dataset}/pair_to_route/{n_bins}/paths.db"

    def test_state_to_node(self):
        with sqlite3.connect(self.route_db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM state_to_node where state=802")
            for row in c.fetchall():
                print(row)

class TestPreProcessGeolifeTest(unittest.TestCase):
    def test_run(self):
        n_bins = 2
        data_dir = "../MTNet_Code/MTNet/data/test/geolife/training_data"
        save_dir = "./test/pair_to_route/geolife_test"
        latlon_config_path = "./dataset_configs/geolife_test.json"
        # make dir of save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        make_pair_to_route.run(n_bins, data_dir, latlon_config_path, save_dir)

    def test_dataset(self):
        n_bins = 2
        dataset = "geolife_test"

        db_path = f"./test/pair_to_route/{dataset}/paths.db"
        # db_path = "/data/chengdu/pair_to_route/30/paths.db"
        data_path = f"./dataset_configs/{dataset}.json"
        with open(data_path, "r") as f:
            configs = json.load(f)
        
        lat_range = configs["lat_range"]
        lon_range = configs["lon_range"]

        print("make grid")
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        grid = Grid(ranges)
        
        # m = folium.Map(location=[30.67, 104.06], zoom_start=12)
        m = folium.Map(location=[39.85, 116.25], zoom_start=12)
        nodes = []
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            for i in range((n_bins+2)**2):
                c.execute(f"SELECT node FROM node_to_state WHERE state={i}")
                node = c.fetchone()
                if node is not None:
                    node = eval(node[0])
                    nodes.append(node)
                    folium.Marker(node, popup=str(i)).add_to(m)
                    latlon = grid.state_to_center_latlon(i)
                    folium.Marker(latlon, popup=str(i), icon=folium.Icon(color='red')).add_to(m)
        m.save(f'./test/data/state_node_{dataset}.html')


class TestPreProcessChengdu(unittest.TestCase):

    def test_make_node_to_state(self):
        DG = make_pair_to_route.make_graph()

        n_bins = 30
        db_path = "./test/data/paths.db"
        # db_path = "/data/chengdu/pair_to_route/30/paths.db"
        data_path = "./dataset_configs/chengdu.json"
        with open(data_path, "r") as f:
            configs = json.load(f)
        lat_range = configs["lat_range"]
        lon_range = configs["lon_range"]
        print("make grid")
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        grid = Grid(ranges)
        n_states = (n_bins+2)**2

        make_pair_to_route.make_node_to_state(DG, n_states, grid.latlon_to_state, db_path)

        # plot states and nodes with anotation and different colors
        with sqlite3.connect(db_path) as conn:
            m = folium.Map(location=[30.67, 104.06], zoom_start=12)
            c = conn.cursor()
            for state in range(32*32):
                c.execute(f"SELECT node FROM node_to_state WHERE state={state}")
                nodes = c.fetchall()
                for node in nodes:
                    node = eval(node[0])
                    folium.Marker(node, popup=str(state)).add_to(m)
        m.save('./test/data/state_node.html')

    def test_process_chengdu(self):
        trajs = make_pair_to_route.load_trajs()
        processed_trajs = make_pair_to_route.process_chengdu(trajs)

        for traj, processed_traj in zip(trajs, processed_trajs):
            self.assertEqual(len(traj), len(processed_traj) - 1)

        # plot test_traj by folium
        test_traj = processed_trajs[1]
        m = folium.Map(location=[30.67, 104.06], zoom_start=12)
        folium.PolyLine(test_traj, color="red", weight=2.5, opacity=1).add_to(m)
        m.save('./test/data/test_traj.html')

    def test_make_paths(self):
        DG, G = make_pair_to_route.make_graph()
        make_pair_to_route.make_paths(G, DG, "./test/data/paths.db")

    def test_make_graph(self):
        DG, G = make_pair_to_route.make_graph()

        # plot nodes
        m = folium.Map(location=[30.67, 104.06], zoom_start=12)
        for node in G.nodes():
            folium.CircleMarker(node, radius=5, color="red", fill=True, fill_color="red").add_to(m)
        m.save('./test/data/nodes.html')


    def test_find_the_node_from_latlon(self):
        DG, G = make_pair_to_route.make_graph()
        latlon = (30.67, 104.06)
        node = make_pair_to_route.find_the_node_from_latlon(DG, latlon)

        # plot latlon and node with different colors
        m = folium.Map(location=[30.67, 104.06], zoom_start=12)
        folium.CircleMarker(latlon, radius=5, color="red", fill=True, fill_color="red").add_to(m)
        folium.CircleMarker(node, radius=5, color="blue", fill=True, fill_color="blue").add_to(m)
        m.save('./test/data/latlon_node.html')

    def test_run(self):
        with open("./config.json", "r") as f:
            latlon_configs = json.load(f)["latlon"]["geolife_test"]
        lat_range = latlon_configs["lat_range"]
        lon_range = latlon_configs["lon_range"]

        make_pair_to_route.run(2, "./test/data", lat_range, lon_range, "./test/data/db")

    def test_db(self):
        dataset = "chengdu"
        n_bins = 14
        db_path = f"/data/{dataset}/pair_to_route/{n_bins}/paths.db"

        with open("./config.json", "r") as f:
            latlon_configs = json.load(f)["latlon"][dataset]
        lat_range = latlon_configs["lat_range"]
        lon_range = latlon_configs["lon_range"]

        print("make grid")
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        grid = Grid(ranges)
        
        
        # count the number of records in state_edge_to_route
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM state_edge_to_route")
            print(c.fetchone()[0], "/", (n_bins+2)**2-n_bins-2)

        start_state = 8
        end_state = 10
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute(f"SELECT route FROM state_edge_to_route WHERE start_state={start_state} AND end_state={end_state}")
            # c.execute(f"SELECT route FROM state_edge_to_route")
            path = c.fetchone()

        if path is not None:
            print(path)
            paths = eval(path[0])
            # plot path
            m = folium.Map(location=[30.67, 104.06], zoom_start=12)
            for i, path in enumerate(paths):
                for j, state in enumerate(path):
                    latlon = grid.state_to_center_latlon(state)
                    # gradiation color
                    color = f"hsl({i*10}, 100%, 50%)"
                    # with anotation
                    folium.CircleMarker(latlon, radius=5, color=color, fill=True, fill_color=color, popup=str(j)).add_to(m)

            m.save('./test/data/state_route.html')
        else:
            print("no path")

    def test_paths(self):
        n_bins = 30
        db_path = "./test/data/paths.db"
        # db_path = "/data/chengdu/pair_to_route/30/paths.db"
        data_path = "./dataset_configs/chengdu.json"
        with open(data_path, "r") as f:
            configs = json.load(f)
        lat_range = configs["lat_range"]
        lon_range = configs["lon_range"]
        print("make grid")
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        grid = Grid(ranges)
        DG = make_pair_to_route.make_graph()

        start_state = 16
        end_state = 2
        start_latlon = grid.state_to_center_latlon(start_state)
        end_latlon = grid.state_to_center_latlon(end_state)
        start_node = make_pair_to_route.find_the_node_from_latlon(DG, start_latlon)
        end_node = make_pair_to_route.find_the_node_from_latlon(DG, end_latlon)

        # find the shortest path from start_node to end_node
        print(start_node, end_node)
        nx_path = nx.shortest_path(DG, start_node, end_node)
        print(nx_path)
        # plot nx_path
        m = folium.Map(location=[30.67, 104.06], zoom_start=12)
        folium.PolyLine(nx_path, color="red", weight=2.5, opacity=1).add_to(m)
        m.save('./test/data/nx_path.html')

        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute(f"SELECT path FROM paths WHERE start_node=? AND end_node=?", (str(start_node), str(end_node)))
            path = eval(c.fetchone()[0])

        # plot path
        m = folium.Map(location=[30.67, 104.06], zoom_start=12)
        folium.PolyLine(path, color="red", weight=2.5, opacity=1).add_to(m)
        m.save('./test/data/path.html')

        for latlon in path:
            print(grid.latlon_to_state(*latlon))


    def test_state_route(self):

        n_bins = 30
        # db_path = "./test/data/paths.db"
        db_path = "/data/chengdu/pair_to_route/30/paths.db"
        data_path = "./dataset_configs/chengdu.json"
        with open(data_path, "r") as f:
            configs = json.load(f)
        lat_range = configs["lat_range"]
        lon_range = configs["lon_range"]
        print("make grid")
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        grid = Grid(ranges)
        
        # count the number of records in state_edge_to_route
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM state_edge_to_route")
            print(c.fetchone()[0])

        start_state = 100
        end_state = 300
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute(f"SELECT route FROM state_edge_to_route WHERE start_state={start_state} AND end_state={end_state}")
            # c.execute(f"SELECT route FROM state_edge_to_route")
            path = c.fetchone()

        
        print(path)
        paths = eval(path[0])
        # plot path
        m = folium.Map(location=[30.67, 104.06], zoom_start=12)
        for i, path in enumerate(paths):
            for j, state in enumerate(path):
                latlon = grid.state_to_center_latlon(state)
                # gradiation color
                color = f"hsl({i*10}, 100%, 50%)"
                # with anotation
                folium.CircleMarker(latlon, radius=5, color=color, fill=True, fill_color=color, popup=str(j)).add_to(m)

        m.save('./test/data/state_route.html')

if __name__ == '__main__':
    unittest.main()
