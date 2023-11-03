import unittest
import sys
import torch
import json
import folium
sys.path.append('./')
from my_utils import construct_default_quadtree
from grid import Grid

class GridTestCase(unittest.TestCase):

    def setUp(self):
        # print current path
        config_path = "./dataset_configs/geolife_narrow.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        self.lat_range = config["lat_range"]
        self.lon_range = config["lon_range"]
        self.n_bins = 30

    def test_grid(self):
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(self.lat_range, self.lon_range, self.n_bins)
        grid = Grid(ranges)

        # plot states with anotation by folium
        n_states = (self.n_bins+2) ** 2
        m = folium.Map(location=[39.9, 116.4], zoom_start=12)
        for i in range(n_states):
            lat, lon = grid.state_to_center_latlon(i)
            folium.Marker([lat, lon], popup=f"{i}").add_to(m)
        m.save('./test/data/test_grid.html')

class QuadTreeTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(QuadTreeTestCase, self).__init__(*args, **kwargs)
        self.n_bins = 6
        self.tree = construct_default_quadtree(self.n_bins)
        self.tree.make_self_complete()

    # generator, optimizer, data_loader = privacy_engine.make_private(module=generator, optimizer=optimizer, data_loader=data_loader, noise_multiplier=args.noise_multiplier, max_grad_norm=args.clipping_bound)
    def test_register_id(self):
        self.tree.register_id()
        assert self.tree.root_node.id == 0
        children = self.tree.root_node.children
        for i in range(4):
            assert children[i].id == i + 1
        
        for i in range(4):
            assert children[0].children[i].id == 5 + i, f"{i+5} != {children[0].children[0].id}"
        
    def test_state_to_path(self):
        assert self.tree.state_to_path(0) == [0, 0, 0]
        assert self.tree.state_to_path(1) == [0, 0, 1]
        assert self.tree.state_to_path(2) == [0, 1, 0]
        assert self.tree.state_to_path(8) == [0, 0, 2]
        assert self.tree.state_to_path(16) == [0, 2, 0]

    def test_make_quad_distribution(self):
        counts = torch.zeros(2, (self.n_bins+2)**2, dtype=torch.float)
        counts[0, 0] = 1
        counts[1, 0] = 0.7
        counts[1, 12] = 0.3
        quad_distribution = self.tree.make_quad_distribution(counts)
        assert quad_distribution.shape == (2, len(self.tree.get_all_nodes()) - len(self.tree.get_leafs()), 4), f"{quad_distribution.shape}"
        assert all(quad_distribution[0, 0] == torch.tensor([1,0,0,0])), f"{quad_distribution[0, 0]}"
        assert all(quad_distribution[1, 0] == torch.tensor([0.7,0.3,0,0])), f"{quad_distribution[1, 0]}"
        assert all(quad_distribution[1, 1] == torch.tensor([1,0,0,0])), f"{quad_distribution[1, 1]}"
        assert all(quad_distribution[1, 2] == torch.tensor([1,0,0,0])), f"{quad_distribution[1, 2]}"
        assert all(quad_distribution[1, 9] == torch.tensor([0,0,1,0])), f"{quad_distribution[1, 3]}"

    def test_make_hidden_ids(self):
        self.assertEqual(self.tree.node_id_to_hidden_id[:10], [0, 0, 1, 2, 3, 4, 5, 8, 9, 6])
    
    def test_node_id_to_hidden_id(self):
        node_id_to_hidden_id = self.tree.node_id_to_hidden_id
        answer = [0, 0, 1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11, 12, 13, 16, 17, 14, 15, 18, 19, 20, 21, 28, 29, 22, 23, 30, 31, 36, 37, 44, 45, 38, 39, 46, 47, 24, 25, 32, 33, 26, 27, 34, 35, 40, 41, 48, 49, 42, 43, 50, 51, 52, 53, 60, 61, 54, 55, 62, 63, 68, 69, 76, 77, 70, 71, 78, 79, 56, 57, 64, 65, 58, 59, 66, 67, 72, 73, 80, 81, 74, 75, 82, 83]
        self.assertEqual(node_id_to_hidden_id[:len(answer)], answer)

    def test_get_location_id_in_the_depth(self):
        self.assertEqual(self.tree.get_location_id_in_the_depth(0, 1), 0)
        self.assertEqual(self.tree.get_location_id_in_the_depth(1, 1), 0)
        self.assertEqual(self.tree.get_location_id_in_the_depth(2, 1), 0)
        self.assertEqual(self.tree.get_location_id_in_the_depth(3, 1), 0)
        self.assertEqual(self.tree.get_location_id_in_the_depth(4, 1), 1)
        self.assertEqual(self.tree.get_location_id_in_the_depth(5, 1), 1)
        self.assertEqual(self.tree.get_location_id_in_the_depth(6, 1), 1)
        self.assertEqual(self.tree.get_location_id_in_the_depth(7, 1), 1)
        self.assertEqual(self.tree.get_location_id_in_the_depth(8, 1), 0)
        self.assertEqual(self.tree.get_location_id_in_the_depth(9, 1), 0)
        self.assertEqual(self.tree.get_location_id_in_the_depth(10, 1), 0)
        self.assertEqual(self.tree.get_location_id_in_the_depth(11, 1), 0)
        self.assertEqual(self.tree.get_location_id_in_the_depth(12, 1), 1)
        self.assertEqual(self.tree.get_location_id_in_the_depth(13, 1), 1)
        self.assertEqual(self.tree.get_location_id_in_the_depth(14, 1), 1)
        self.assertEqual(self.tree.get_location_id_in_the_depth(15, 1), 1)
        self.assertEqual(self.tree.get_location_id_in_the_depth(16, 1), 0)
        self.assertEqual(self.tree.get_location_id_in_the_depth(17, 1), 0)
        self.assertEqual(self.tree.get_location_id_in_the_depth(18, 1), 0)
        self.assertEqual(self.tree.get_location_id_in_the_depth(19, 1), 0)
        self.assertEqual(self.tree.get_location_id_in_the_depth(20, 1), 1)
        self.assertEqual(self.tree.get_location_id_in_the_depth(21, 1), 1)
        self.assertEqual(self.tree.get_location_id_in_the_depth(22, 1), 1)
        self.assertEqual(self.tree.get_location_id_in_the_depth(23, 1), 1)

if __name__ == "__main__":
    unittest.main()