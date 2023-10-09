import unittest
import torch
import numpy as np 
# add parent path
import sys
sys.path.append('./')
# from run_ import compute_distribution_js_for_each_depth, make_target_distributions_of_all_layers, make_second_order_test_data
from run import make_second_order_test_data
from my_utils import construct_default_quadtree

class TestRun(unittest.TestCase):
    def test_make_second_order_test_data(self):
        dataset_name = "peopleflow"
        top_second_order_base_locations = [(0,1), (0,2), (1,0)]
        test_traj, test_traj_time = make_second_order_test_data(top_second_order_base_locations, dataset_name)
        self.assertEqual(test_traj[0], [0,1,3])
        self.assertEqual(test_traj[1], [0,2,3])
        self.assertEqual(test_traj[2], [1,0,3])


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