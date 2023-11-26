import numpy as np
import argparse
import config.folder_and_file_names as fname
import pathlib
import json
import sys


sys.path.append("../../")
from my_utils import get_original_dataset_name

class ParSetter:

    def __init__(self):
        pass

    def set_up_args(self, dataset_file_name=None, epsilon=False, epsilon_partition=False, level1_parameter=False, level2_parameter=False):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default="chengdu")
        parser.add_argument('--data_name', type=str, default="10000")
        parser.add_argument('--training_data_name', type=str, default="privtrace")
        parser.add_argument('--dataset_config_path', type=str, default="../../config.json")
        parser.add_argument('--subdividing_inner_parameter', type=float, default=200)
        parser.add_argument('--total_epsilon', type=float, default=1)
        # regularly, partition solution is suggested to be np.array([0.2, 0.52, 0.28]))
        parser.add_argument('--epsilon_partition', type=np.ndarray, default=[0.2, 0.4, 0.4])
        # this parameter indicates how many trajectories to generate
        parser.add_argument('--trajectory_number_to_generate', type=int, default=-1)
        parser.add_argument('--fixed_divide_parameter', type=int, default=0)
        parser.add_argument('--save_name', type=str)
        # this is for our evaluation <- Shun
        parser.add_argument('--n_bins', type=int, default=30)
        args = vars(parser.parse_args())

        if epsilon is not False:
            args['total_epsilon'] = epsilon
        if epsilon_partition is not False:
            args['epsilon_partition'] = epsilon_partition
        if level1_parameter is not False:
            args['level1_divide_inner_parameter'] = level1_parameter
        if level2_parameter is not False:
            args['subdividing_inner_parameter'] = level2_parameter
        if dataset_file_name is not None:
            args['dataset_file_name'] = dataset_file_name

        print(f"using the range of", args["dataset_config_path"])
        with open(args["dataset_config_path"], "r") as f:
            params = json.load(f)["latlon"][get_original_dataset_name(args["dataset"])]
        args['lat_range'] = params['lat_range']
        args['lon_range'] = params['lon_range']
        # args['dataset'] = args["dataset"]
        # args['data_name'] = args["data_name"]
        return args




