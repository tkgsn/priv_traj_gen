import unittest
import torch
import numpy as np 
from mocks import GeneratorMock, NameSpace
# add parent path
import sys
sys.path.append('./')
# from run_ import compute_distribution_js_for_each_depth, make_target_distributions_of_all_layers, make_second_order_test_data
from run import evaluate
from my_utils import construct_default_quadtree, set_logger
from dataset import TrajectoryDataset
import pathlib

def make_data():
    n_data = 100
    n_locations = 16
    n_split = 5
    pad = np.ones((int(n_data/2), 1), dtype=int)
    traj1 = np.concatenate([np.zeros((int(n_data/2), 1)).astype(int), pad, 2*np.ones((int(n_data/2), 1)).astype(int)], axis=1).tolist()
    traj2 = np.concatenate([np.zeros((int(n_data/2), 1)).astype(int), pad, 6*np.ones((int(n_data/2), 1)).astype(int)], axis=1).tolist()
    traj = np.concatenate([traj1, traj2], axis=0).tolist()
    traj_time = [[0, 1, 2]]*n_data


    traj1 = np.concatenate([np.zeros((int(n_data/2), 1)).astype(int), pad, 2*np.ones((int(n_data/2), 1)).astype(int)], axis=1).tolist()
    traj2 = np.concatenate([np.zeros((int(n_data/2), 1)).astype(int), pad, 3*np.ones((int(n_data/2), 1)).astype(int), 6*np.ones((int(n_data/2), 1)).astype(int)], axis=1).tolist()
    route_traj = traj1
    route_traj.extend(traj2)

    return TrajectoryDataset(traj, traj_time, n_locations, n_split, dataset_name="test", route_data=route_traj)

def make_args():
    args = NameSpace()
    args.eval_interval = 1
    args.n_test_locations = 2
    args.dataset = "peopleflow"
    args.n_split = 5
    args.batch_size = 10
    args.save_path = "./test/test"
    args.remove_first_value = True
    args.evaluate_first_next_location = True
    args.evaluate_second_next_location = True
    args.evaluate_second_order_next_location = True
    args.evaluate_global = True
    args.evaluate_passing = True
    args.evaluate_source = True
    args.evaluate_target = True
    args.evaluate_route = True
    args.evaluate_destination = True
    args.evaluate_distance = True
    return args

class TestRun(unittest.TestCase):
    # def test_make_second_order_test_data(self):
    #     dataset_name = "peopleflow"
    #     top_second_order_base_locations = [(0,1), (0,2), (1,0)]
    #     test_traj, test_traj_time = make_second_order_test_data(top_second_order_base_locations, dataset_name)
    #     self.assertEqual(test_traj[0], [0,1,3])
    #     self.assertEqual(test_traj[1], [0,2,3])
    #     self.assertEqual(test_traj[2], [1,0,3])


    # def test_compute_distribution_js_for_each_depth(self):
    #     generated_next_location_distribution = torch.zeros(3, 16)
    #     next_location_distribution = torch.zeros(3, 16)
    #     generated_next_location_distribution[0,0] = 0.5
    #     generated_next_location_distribution[0,1] = 0.5
    #     next_location_distribution[0,0] = 1

    #     generated_next_location_distribution[1,0] = 1
    #     next_location_distribution[1,0] = 1

    #     generated_next_location_distribution[2,0] = 1/3
    #     generated_next_location_distribution[2,1] = 1/3
    #     generated_next_location_distribution[2,2] = 1/3
    #     next_location_distribution[2,0] = 1
    #     a = compute_distribution_js_for_each_depth(generated_next_location_distribution, next_location_distribution)
    #     self.assertTrue(np.allclose(a, np.array([[0,0.21576156],[0,0],[0.1323041,0.31825706]]), atol=1e-05))

    # def test_make_target_distributions_of_all_layers(self):
    #     n_bins = 6
    #     n_locations = (n_bins+2)**2
    #     tree = construct_default_quadtree(n_bins)
    #     tree.make_self_complete()
    #     target_distribution = torch.zeros(3, n_locations)
    #     target_distribution[0,10] = 1
    #     target_distribution[1,1] = 1
    #     target_distribution[2,45] = 1
    #     made_distributions = make_target_distributions_of_all_layers(target_distribution, tree)
    #     # depth = 0
    #     self.assertTrue(made_distributions[0][0][0] == 1)
    #     self.assertTrue(made_distributions[0][1][0] == 1)
    #     self.assertTrue(made_distributions[0][2][3] == 1)
    #     # depth = 1
    #     self.assertTrue(made_distributions[1][0][1] == 1)
    #     self.assertTrue(made_distributions[1][1][0] == 1)
    #     self.assertTrue(made_distributions[1][2][10] == 1)
    #     # depth = 2
    #     self.assertTrue(made_distributions[2][0][10] == 1)
    #     self.assertTrue(made_distributions[2][1][1] == 1)
    #     self.assertTrue(made_distributions[2][2][45] == 1)

    #     target_distribution = torch.zeros(1, n_locations)
    #     target_distribution[0,10] = 1/2
    #     target_distribution[0,4] = 1/2
    #     made_distributions = make_target_distributions_of_all_layers(target_distribution, tree)
    #     self.assertTrue(made_distributions[0][0][0] == 1/2)
    #     self.assertTrue(made_distributions[0][0][1] == 1/2)
    #     self.assertTrue(made_distributions[1][0][1] == 1/2)
    #     self.assertTrue(made_distributions[1][0][2] == 1/2)
    #     self.assertTrue(made_distributions[2][0][10] == 1/2)
    #     self.assertTrue(made_distributions[2][0][4] == 1/2)
        
if __name__ == '__main__':
    unittest.main()