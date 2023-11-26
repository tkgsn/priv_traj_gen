import unittest
import folium
import config.folder_and_file_names as fname
from discretization.get_discretization import DisData
from primarkov.build_markov_model import ModelBuilder
from generator.state_trajectory_generation import StateGeneration
from generator.to_real_translator import RealLocationTranslator
# from tools.object_store import ObjectStore
from config.parameter_carrier import ParameterCarrier
from config.parameter_setter import ParSetter
from tools.data_writer import DataWriter
from data_preparation.data_preparer import DataPreparer
import datetime
import pandas as pd
import config.folder_and_file_names as config
import pathlib
import json
import pickle
import privtrace_generator
import sys
import numpy as np

sys.path.append("../priv_traj_gen")
from grid import Grid
"""
about usable_index:
usable_index seems to be always the same as the index of the level2_cell, that is, trajectory.has_not_usable_index is always true
"""



class TestPrivTrace(unittest.TestCase):

    def setUp(self):
        print("methodname", self._testMethodName)
        self.par = ParSetter().set_up_args()
        self.par["total_epsilon"] = 0
        self.pc = ParameterCarrier(self.par)
    
    def test_1_get_trajectory_set(self):
        """
        trajectory_set has trajectory_list that is a list of trajectory_array
        trajectory_array is a numpy array of shape (d, 2) where d is the seq_len and 2 is the lat and lon
        """
        data_preparer = DataPreparer(self.par)
        trajectory_set = data_preparer.get_trajectory_set(self.pc)
        # save by pickle
        with open("./data/trajectory_set.pickle", "wb") as f:
            pickle.dump(trajectory_set, f)
        print(trajectory_set.trajectory_list[0].trajectory_array.shape)
        print(trajectory_set.trajectory_list[0].trajectory_array)

    def test_2_disdata(self):
        """
        grid is made according to trajectory_set; grid has two layers if a density for a above cell is above the threshold, then it is divided into subcells
        our modification restricts that the number of subcells is a power of 2 for fair comparison
        Also, this converts a (lat,lon) trajectory to the (index) trajectory
        the information is saved to trajectory_set
        maybe, also computing neighbor of each cell
        """
        # load trajectory set
        with open("./data/trajectory_set.pickle", "rb") as f:
            trajectory_set = pickle.load(f)
        disdata1 = DisData(self.pc)
        grid = disdata1.get_discrete_data(trajectory_set)
        # save grid by pickle
        with open("./data/grid.pickle", "wb") as f:
            pickle.dump(grid, f)

        # save trajectory_set by pickle
        with open("./data/trajectory_set_gridded.pickle", "wb") as f:
            pickle.dump(trajectory_set, f)

    def test_3_model_building(self):
        """
        just computing the transition probability based of trajectory_set and grid
        note that a contribution of each trajectory is 1 by dividing it by the seq_len to guarantee the sensitivity 1 
        and add noise to it
        also, this computes the neighboring_matrix based on grid
        note that start_state = n_vocab, end_state = n_vocab + 1, 
        """
        # load trajectory_set and grid
        with open("./data/trajectory_set_gridded.pickle", "rb") as f:
            trajectory_set = pickle.load(f)
        with open("./data/grid.pickle", "rb") as f:
            grid = pickle.load(f)

        mb1 = ModelBuilder(self.pc)
        mo1 = mb1.build_model(grid, trajectory_set)

        # save model by pickle
        with open("./data/model.pickle", "wb") as f:
            pickle.dump(mo1, f)

    def test_4_model_filtering(self):
        """
        start_end_trip_distribution_calibration computes the distribution of start locations
        mb1.get_sensitive_state gives the inforamtion for model selection?
        guidepost supports the generation by 2nd order markov

        """

        # load trajectory_set and grid, model
        with open("./data/trajectory_set_gridded.pickle", "rb") as f:
            trajectory_set = pickle.load(f)
        with open("./data/grid.pickle", "rb") as f:
            grid = pickle.load(f)
        with open("./data/model.pickle", "rb") as f:
            mo1 = pickle.load(f)
        
        mb1 = ModelBuilder(self.pc)
        mo1 = mb1.filter_model(trajectory_set, grid, mo1)

        with open("./data/model_filtered.pickle", "wb") as f:
            pickle.dump(mo1, f)

    def test_5_generate_tra(self):

        self.pc.trajectory_number_to_generate = 10000
        # note that the trajectory_number_to_generate is -1 by default, which means the same numebr of trajectories as the original dataset by DisData
        sg1 = StateGeneration(self.pc)

        # load model
        with open("./data/model_filtered.pickle", "rb") as f:
            mo1 = pickle.load(f)
        st_tra_list = sg1.generate_tra(mo1, False)

        # trajectory should not have a same state twice in a row
        for traj in st_tra_list:
            for i in range(len(traj)-1):
                self.assertNotEqual(traj[i], traj[i+1])



class TestPrivTraceGenerator(unittest.TestCase):

    def setUp(self):
        print("methodname", self._testMethodName)
        self.par = ParSetter().set_up_args()
        self.pc = ParameterCarrier(self.par)

        self.n_bins = 30
        self.lat_range = self.par["lat_range"]
        self.lon_range = self.par["lon_range"]
        self.ranges = Grid.make_ranges_from_latlon_range_and_nbins(self.lat_range, self.lon_range, self.n_bins)
        self.our_grid = Grid(self.ranges)

    def test_1_make_privtrace_id_to_states(self):

        lat_range = self.par["lat_range"]
        lon_range = self.par["lon_range"]
        n_bins = 30
        privtrace_id_to_states = privtrace_generator.make_privtrace_id_to_states(lat_range, lon_range, n_bins)

        # save by json
        with open("./data/privtrace_id_to_states.json", "w") as f:
            json.dump(privtrace_id_to_states, f)

    def test_2_generation(self):

        # load model and privtrace_id_to_states
        with open("./data/model_filtered.pickle", "rb") as f:
            privtrace_model = pickle.load(f)
        with open("./data/privtrace_id_to_states.json", "r") as f:
            privtrace_id_to_states = json.load(f)
            # convert to int
            privtrace_id_to_states = {int(k): v for k, v in privtrace_id_to_states.items()}

        generator = privtrace_generator.PrivTraceGenerator(self.pc, privtrace_model, privtrace_id_to_states)
        
        references = np.zeros(((self.n_bins+2)**2, 3), dtype=int)
        references[:, 0] = np.arange((self.n_bins+2)**2)
        mini_batch_size = 1
        trajs = generator.make_sample(references, mini_batch_size)
        self.assertEqual(len(trajs), len(references))
        for traj, reference in zip(trajs, references):
            self.assertEqual(traj[0], reference[0])
            # trajectory should not have a same state twice in a row
            for i in range(len(traj)-1):
                self.assertNotEqual(traj[i], traj[i+1])


        # plot traj by folium
        m = folium.Map(location=[self.lat_range[0], self.lon_range[0]], zoom_start=10)
        for i, state in enumerate(trajs[0]):
            lat,lon = self.our_grid.state_to_center_latlon(state)
            folium.Marker(location=[lat, lon], popup=f"{i}", icon=folium.Icon(color='green')).add_to(m)
        m.save("./data/traj.html")


    
    def test_privtrace_id_and_states(self):
        with open("./data/grid.pickle", "rb") as f:
            grid = pickle.load(f)

        lat_max, lat_min, lon_min, lon_max = grid.level2_borders[0]
        # plot border_latlons by folium
        m = folium.Map(location=[lat_max, lon_max], zoom_start=10)
        border_latlons = grid.level2_borders
        for border_latlon in border_latlons:
            lat_max, lat_min, lon_min, lon_max = border_latlon
            folium.Rectangle(bounds=[(lat_min, lon_min), (lat_max, lon_max)], color='red', fill=True, fill_color='red', fill_opacity=0.1).add_to(m)
        central_points_gps = grid.usable_state_central_points()
        for i, (lat, lon) in enumerate(central_points_gps):
            folium.Marker(location=[lat, lon], popup=f"{i}", icon=folium.Icon(color='green')).add_to(m)

        # the order of central_points_gps is the same as the order of privtrace_id, that is, border_latlons
        m.save("./data/border_latlons_privtrace.html")

        lat_range = self.par["lat_range"]
        lon_range = self.par["lon_range"]
        n_bins = 30
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        m = folium.Map(location=[lat_max, lon_max], zoom_start=10)
        for i, (lon, lat) in enumerate(ranges):
            left_lon, right_lon = lon
            top_lat, bottom_lat = lat
            folium.Rectangle(bounds=[(bottom_lat, left_lon), (top_lat, right_lon)], color='red', fill=True, fill_color='red', fill_opacity=0.1).add_to(m)
            # plot anotation
            folium.Marker(location=[(bottom_lat+top_lat)/2, (left_lon+right_lon)/2], popup=f"{i}", icon=folium.Icon(color='green')).add_to(m)

        m.save("./data/border_latlons_ours.html")



if __name__ == '__main__':
    unittest.main()