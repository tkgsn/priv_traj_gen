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
    def setUp(self):
        print(self._testMethodName)
        self.n_locations = 16
        traj_type_dim = 1
        hidden_dim = 30
        batch_size = 10
        n_data = 120
        location_embedding_dim = 10
        self.n_split = 5
        time_dim = self.n_split+3
    
    # def test_compute_auxiliary_information(self):
    #     dataset = TrajectoryDataset(self.traj, self.traj_time, self.n_locations, self.n_split)
    #     dataset.compute_auxiliary_information(pathlib.Path("./test/test/test/test_run.log"), self.logger)

    #     self.assertEqual(dataset.second_order_next_location_counts[(0, 1)][2], 50)
    #     self.assertEqual(dataset.second_order_next_location_counts[(0, 1)][3], 50)
    #     self.assertEqual(dataset.next_location_counts[0][1], 100)
    #     self.assertEqual(dataset.next_location_counts[1][2], 50)
    #     self.assertEqual(dataset.next_location_counts[1][3], 50)
    #     self.assertEqual(dataset.first_next_location_counts[0][1], 100)
    #     self.assertEqual(dataset.second_next_location_counts[1][2], 50)
    #     self.assertEqual(dataset.second_next_location_counts[1][3], 50)

    # def test_real_start(self):
    #     dataset = TrajectoryDataset(self.traj, self.traj_time, self.n_locations, self.n_split, real_start=False)
        

    def test_make_test_data_loader(self):

        n_data = 120
        traj1 = np.concatenate([np.zeros((int(n_data/4), 1)).astype(int), np.ones((int(n_data/4), 1)).astype(int), 2*np.ones((int(n_data/4), 1)).astype(int)], axis=1).tolist()
        traj2 = np.concatenate([np.zeros((int(n_data/4), 1)).astype(int), 2*np.ones((int(n_data/4), 1)).astype(int), 3*np.ones((int(n_data/4), 1)).astype(int)], axis=1).tolist()
        traj3 = np.concatenate([np.zeros((int(n_data/4), 1)).astype(int), np.ones((int(n_data/4), 1)).astype(int), 4*np.ones((int(n_data/4), 1)).astype(int)], axis=1).tolist()
        traj4 = np.concatenate([5*np.ones((int(n_data/4), 1)).astype(int), np.ones((int(n_data/4), 1)).astype(int), 4*np.ones((int(n_data/4), 1)).astype(int)], axis=1).tolist()
        self.traj = np.concatenate([traj1, traj2, traj3, traj4], axis=0).tolist()
        self.traj_time = [[0, 1, 2]]*n_data
        self.logger = set_logger(__name__, pathlib.Path("./test/test_dataset.log"))

        dataset = TrajectoryDataset(self.traj, self.traj_time, self.n_locations, self.n_split)
        dataset.compute_auxiliary_information(pathlib.Path("./test/test/test/test_run.log"), self.logger)

        test_data_loader, counters = dataset.make_first_order_test_data_loader(1)
        self.assertEqual(len(counters), 1)
        self.assertEqual(counters[0], 90)

        test_data_loader, counters = dataset.make_first_order_test_data_loader(5)
        self.assertEqual(counters[0], 90)
        self.assertEqual(counters[5], 30)

        trajs = []
        for mini_batch in test_data_loader:
            trajs.extend(mini_batch["input"].tolist())
        self.assertEquals([traj[1:] for traj in trajs], self.traj)

        test_data_loader, counters = dataset.make_second_order_test_data_loader(5)
        self.assertEqual(counters[(0,1)], 60)
        self.assertEqual(counters[(0,2)], 30)
        self.assertEqual(counters[(5,1)], 30)

        trajs = []
        for mini_batch in test_data_loader:
            trajs.extend(mini_batch["input"].tolist())
        self.assertEqual(trajs[30], [self.n_locations, 0, 1, 4])



if __name__ == "__main__":
    unittest.main()