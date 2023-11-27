import sys
import pathlib
import pandas as pd
import json
from logging import getLogger
import pickle

import matplotlib.pyplot as plt

from config.parameter_carrier import ParameterCarrier
from config.parameter_setter import ParSetter
from data_preparation.data_preparer import DataPreparer
from discretization.get_discretization import DisData
from primarkov.build_markov_model import ModelBuilder
import privtrace_generator

sys.path.append("../../")
from my_utils import load
from dataset import TrajectoryDataset
import evaluation

def set_args():

    class Namespace():
        pass

    args = Namespace()
    args.evaluate_global = False
    args.evaluate_passing = True
    args.evaluate_source = True
    args.evaluate_target = True
    args.evaluate_route = True
    args.evaluate_destination = True
    args.evaluate_distance = True
    args.evaluate_first_next_location = False
    args.evaluate_second_next_location = False
    args.evaluate_second_order_next_location = False
    args.eval_initial = True
    args.eval_interval = 1
    args.compensation = True
    args.n_test_locations = 30
    args.dataset = "chengdu"
    args.n_split = 5
    # this is not used
    args.batch_size = 100

    return args

if __name__ == "__main__":

    training_data_dir = pathlib.Path(sys.argv[1])
    route_data_dir = pathlib.Path(sys.argv[2])
    save_dir = pathlib.Path(sys.argv[3])
    epsilon = float(sys.argv[4])
    # k = int(sys.argv[4])
    # epsilon = float(sys.argv[5])

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "imgs").mkdir(parents=True, exist_ok=True)


    print("load data from", training_data_dir)
    print("load route data from", route_data_dir)
    training_data = load(training_data_dir / "training_data.csv")
    time_data = load(training_data_dir / "training_data_time.csv")
    route_data = load(training_data_dir / "route_training_data.csv")
    gps = pd.read_csv(training_data_dir / "gps.csv", header=None).values

    print("load from", save_dir / f"model_{epsilon}" / f"privtrace_generator.pickle")
    with open(save_dir / f"model_{epsilon}" / f"privtrace_generator.pickle", "rb") as f:
        generator = pickle.load(f)

    # load setting file
    with open(training_data_dir / "params.json", "r") as f:
        param = json.load(f)
    n_bins = param["n_bins"]
    n_locations = (n_bins+2)**2

    args = set_args()
    args.save_path = str(save_dir)

    # for evaluation, we use stay_point_trajectories with route_data
    dataset = TrajectoryDataset(training_data, time_data, n_locations, args.n_split, route_data=route_data)
    # dataset.compute_auxiliary_information(save_dir, getLogger(__name__))
    print(generator.make_sample(dataset.references[0:100], None))
    
    # results = [evaluation.run(generator, dataset, args, 0)]

    # print("save results to", save_dir / "params.json")
    # with open(save_dir / "params.json", "w") as f:
        # json.dump(results, f)