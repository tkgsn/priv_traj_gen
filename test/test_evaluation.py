import unittest

# add parent path
import sys
import numpy as np
import pathlib
from unittest.mock import MagicMock
import torch
sys.path.append('./')
from evaluation import evaluate_next_location_on_test_dataset
from my_utils import set_logger
from run import make_second_order_test_data
from dataset import TrajectoryDataset

class EvaluationTestCase(unittest.TestCase):
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
    
    def test_evaluate_next_location_on_test_dataset(self):
        dataset_name = "peopleflow"
        top_second_order_base_locations = [(0,1), (0,2), (1,0)]
        test_traj, test_traj_time = make_second_order_test_data(top_second_order_base_locations, dataset_name)
        test_dataset = TrajectoryDataset(test_traj, test_traj_time, 16, 5)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, collate_fn=test_dataset.make_padded_collate(True))
    
        next_location_distributions_ = torch.zeros(3, 16)
        next_location_distributions_[0, 0] = 1
        next_location_distributions_[1, 1] = 1
        next_location_distributions_[2, 2] = 1
        next_location_distributions = {key: value for key, value in zip(top_second_order_base_locations, next_location_distributions_)}

        time = None
        # seq_len = 3
        location_output = torch.log(next_location_distributions_.repeat(1,3)).view(-1, 3, 16)

        class GeneratorMock():

            def __call__(self, *args, **kwargs):
                return location_output, time

            def parameters(self):
                # make the empty object that has device attribute
                class Empty():
                    pass
                a = Empty()
                a.device = torch.device("cpu")
                yield a
    
        generator_mock = GeneratorMock()

        jss = evaluate_next_location_on_test_dataset(next_location_distributions, test_data_loader, generator_mock, 2, 2)
        # all values should be 0
        self.assertTrue(np.allclose(jss, np.zeros((3,2)), atol=1e-05))

if __name__ == "__main__":
    unittest.main()