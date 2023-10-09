import unittest

# add parent path
import sys
import numpy as np
import pathlib
import json
sys.path.append('./')
from dataset import TrajectoryDataset
from my_utils import set_logger

class TrajectoryDatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        n_locations = 16
        traj_type_dim = 1
        hidden_dim = 30
        batch_size = 10
        n_data = 100
        location_embedding_dim = 10
        n_split = 5
        time_dim = n_split+3

        pad = np.ones((int(n_data/2), 1), dtype=int)
        traj1 = np.concatenate([np.zeros((int(n_data/2), 1)).astype(int), pad, 2*np.ones((int(n_data/2), 1)).astype(int)], axis=1).tolist()
        traj2 = np.concatenate([np.zeros((int(n_data/2), 1)).astype(int), pad, 3*np.ones((int(n_data/2), 1)).astype(int)], axis=1).tolist()
        traj = np.concatenate([traj1, traj2], axis=0)
        traj_time = [[0, 1, 2]]*n_data
        self.dataset = TrajectoryDataset(traj, traj_time, n_locations, n_split)
        self.logger = set_logger(__name__, pathlib.Path("./test/test_dataset.log"))
    
    def test_compute_auxiliary_information(self):
        save_path = pathlib.Path("./test/_")
        self.dataset.compute_auxiliary_information(pathlib.Path("./test/_"), self.logger)
        next_location_count_path = save_path.parent / f"0_second_order_next_location_count.json"
        with open(next_location_count_path) as f:
            next_location_counts = json.load(f)
            next_location_counts = {eval(key): value for key, value in next_location_counts.items()}
        
        self.assertEqual(next_location_counts[(0, 1)][2], 50)
        self.assertEqual(next_location_counts[(0, 1)][3], 50)

if __name__ == "__main__":
    unittest.main()