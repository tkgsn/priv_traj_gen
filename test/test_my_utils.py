import unittest

# add parent path
import sys
sys.path.append('./')
from my_utils import save, load, set_budget, depth_clustering, plot_density

class DataPreProcessingTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DataPreProcessingTestCase, self).__init__(*args, **kwargs)

    def test_save_load(self):
        data = [[1,2,3], [4,5,6], [7,8,9]]
        save('test_data', data)
        data_loaded = load('test_data')
        self.assertEqual(data, data_loaded)
        data_loaded = load('test_data', 2)
        print(data_loaded)
        self.assertEqual(2, len(data_loaded))

        data = [[[0, 124.6, 30.45], [1, 124.6, 30.45]], [[0, 124.6, 30.45], [1, 124.6, 30.45], [2, 31.23, 23.34]]]
        save('test_data', data)
        data_loaded = load('test_data')
        self.assertEqual(data, data_loaded)

    def test_set_budget(self):
        set_budget(10000, 62)

    def test_depth_clustering(self):
        # class needs to correspond to node id
        n_bins = 14
        location_to_class, quad_tree = depth_clustering(n_bins)
        self.assertEqual(len(set(location_to_class.keys())), (n_bins+2)**2)
        self.assertEqual(len(set(location_to_class.values())), 16)
        self.assertEqual(len(quad_tree.get_leafs()), 16)
        self.assertEqual(location_to_class[8], 4)

    def test_plot_density(self):
        plot_density([0,1,2,3,4,5,6,7,8], 9, "./test/data/test.png", 6)

if __name__ == "__main__":
    unittest.main()