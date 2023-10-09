import unittest

# add parent path
import sys
sys.path.append('./')
from make_raw_data import make_raw_data_chengdu, make_raw_data_rotation, make_raw_data_random
from my_utils import construct_default_quadtree

class MakeChengduRawDataTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MakeChengduRawDataTestCase, self).__init__(*args, **kwargs)
    
    def test_make_chengdu_raw_data(self):
        trajs = make_raw_data_chengdu()
        # print(trajs[0])

class MakeRotationRawDataTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MakeRotationRawDataTestCase, self).__init__(*args, **kwargs)
    
    def test_make_rotation_raw_data(self):
        # a rotation trajectoary has always 2 points
        # the start location is always a location whose id is an even number in the second layer
        # the end location is always the location whose id is the next odd number in the second layer
        n_bins = 14
        tree = construct_default_quadtree(n_bins)
        tree.make_self_complete()
        trajs = make_raw_data_rotation(0, 100, n_bins)
        print(trajs)
        for traj in trajs:
            start_location = traj[0]
            start_location_id = tree.get_location_id_in_the_depth(start_location, 2)
            end_location = traj[1]
            end_location_id = tree.get_location_id_in_the_depth(end_location, 2)
            self.assertEqual(start_location_id%2, 0)
            self.assertEqual(end_location_id, start_location_id+1)
        # print(trajs[0])

class MakeRandomRawDataTestCase(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
    
    def test_make_random_raw_data(self):
        trajs = make_raw_data_random(0, 100, 14)


if __name__ == "__main__":
    unittest.main()