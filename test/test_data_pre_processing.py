import unittest
from unittest.mock import patch
import sys
import folium

sys.path.append('./')
import data_pre_processing
from grid import Grid
from my_utils import load, load_latlon_range
from evaluation import compensate_trajs

class TestDataPreProcessing(unittest.TestCase):

    def setUp(self):
        # Set up any necessary test data
        pass

    def test_make_reversible_stay_traj(self):
        trajs = []
        with open("./test/data/chengdu_trajs.txt", "r") as f:
            for line in f:
                trajs.append(list(map(int, line.strip().split(","))))
        
        reversible_stay_trajs = []
        for traj in trajs:
            road_db = "/data/chengdu/pair_to_route/30_tr0/paths.db"
            reversible_stay_traj = data_pre_processing.make_reversible_stay_traj(traj, road_db)
            reversible_stay_trajs.append(reversible_stay_traj)
        
        reversed_stay_trajs, _ = compensate_trajs(reversible_stay_trajs, road_db)
        for i in range(len(trajs)):
            self.assertEqual(trajs[i], reversed_stay_trajs[i])

    def test_geolife_dataset(self):
        dataset = "geolife_test"
        n_bins = 30
        location_threshold = 200
        time_threshold = 10
        mm_data_path = f"/data/{dataset}_mm/0/{location_threshold}_{time_threshold}_bin{n_bins}_seed0/training_data.csv"
        data_path = f"/data/{dataset}/0/{location_threshold}_{time_threshold}_bin{n_bins}_seed0/training_data.csv"

        mm_data = load(mm_data_path)
        data = load(data_path)

        lat_range, lon_range = load_latlon_range(dataset)

        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        grid = Grid(ranges)

        # plot traj by folium
        m = folium.Map(location=[lat_range[0], lon_range[0]], zoom_start=12)
        for i in range(len(mm_data)):
            traj = mm_data[i]
            traj = [grid.state_to_random_latlon_in_the_cell(state) for state in traj]
            folium.PolyLine(traj, color="blue", weight=2.5, opacity=1).add_to(m)
        m.save(f"./test/data/{dataset}_mm_training_data.html")

        m = folium.Map(location=[lat_range[0], lon_range[0]], zoom_start=12)
        for i in range(len(data)):
            traj = data[i]
            traj = [grid.state_to_random_latlon_in_the_cell(state) for state in traj]
            folium.PolyLine(traj, color="blue", weight=2.5, opacity=1).add_to(m)
        m.save(f"./test/data/{dataset}_training_data.html")





if __name__ == '__main__':
    unittest.main()