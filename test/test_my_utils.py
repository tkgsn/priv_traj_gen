import unittest

# add parent path
import sys
sys.path.append('./')
from my_utils import save, load

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

if __name__ == "__main__":
    unittest.main()