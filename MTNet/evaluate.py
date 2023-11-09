# evaluation using evaluate function in ../../priv_traj_gen/run.py
import sys
import pathlib
from logging import getLogger
import json
import numpy as np
sys.path.append('../../priv_traj_gen')
import evaluation
from my_utils import load
from dataset import TrajectoryDataset
import tqdm

class MockGenerator():
    '''
    This generator returns the trajectory in the given path
    '''
    def __init__(self, path, random=False):
        # load
        self.trajs = load(path)
        self.cursor = 0
        self.random = random

    def eval(self):
        pass

    def train(self):
        pass

    def make_sample(self, references, mini_batch_size):
        '''
        return the mini_batch_size trajectories in the given path
        '''
        if self.random:
            if len(self.trajs)-mini_batch_size == 0:
                self.cursor = 0
            else:
                self.cursor = np.random.randint(0, len(self.trajs)-mini_batch_size)
        else:
            self.cursor += mini_batch_size
            if self.cursor >= len(self.trajs)+mini_batch_size:
                self.cursor = mini_batch_size
        return self.trajs[self.cursor-mini_batch_size:self.cursor]

class Namespace():
    pass

def set_args():

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
    args.compensation = False
    args.eval_initial = True
    args.eval_interval = 1
    args.n_test_locations = 30
    args.dataset = "chengdu"
    args.n_split = 5
    # this is not used
    args.batch_size = 100

    return args

def run(generated_data_path, original_training_data_path, stay_point_data_path, save_path):

    generated_data_path = pathlib.Path(generated_data_path)
    original_training_data_path = pathlib.Path(original_training_data_path)
    stay_point_data_path = pathlib.Path(stay_point_data_path)
    save_path = pathlib.Path(save_path)
    (save_path / "imgs").mkdir(parents=True, exist_ok=True)

    logger = getLogger(__name__)
    trajectories = load(original_training_data_path / "training_data.csv")
    stay_point_trajectories = load(stay_point_data_path / "training_data.csv")
    time_trajectories = load(stay_point_data_path / "training_data_time.csv")
    print(f"load training data from {original_training_data_path / 'training_data.csv'}")
    print(f"load stay point data from {stay_point_data_path / 'training_data.csv'}")
    print(f"load time data from {original_training_data_path / 'training_data_time.csv'}")

    if len(trajectories) != len(stay_point_trajectories) or len(trajectories) != len(time_trajectories):
        print("WARNING: DATA SIZES SHOULD BE THE SAME")
    # print(len(trajectories), len(stay_point_trajectories), len(time_trajectories))

    args = set_args()
    args.save_path = str(save_path)

    # load setting file
    with open(original_training_data_path / "params.json", "r") as f:
        param = json.load(f)
    n_bins = param["n_bins"]
    n_locations = (n_bins+2)**2

    # for evaluation, we use stay_point_trajectories with route_data
    dataset = TrajectoryDataset(stay_point_trajectories, time_trajectories, n_locations, args.n_split, route_data=trajectories)
    dataset.compute_auxiliary_information(save_path, logger)

    # find the generated data in the given generated_data_path and sort
    files = [file for file in generated_data_path.iterdir() if file.name.startswith("generated_")]
    files.sort(key=lambda x: int(x.name.split("_")[1].split(".")[0]))

    # evaluation
    resultss = []
    for i, file in tqdm.tqdm(enumerate(files)):
        print(f"evaluate {file}")
        generator = MockGenerator(file)
        epoch = i
        results = evaluation.run(generator, dataset, args, epoch)
        print(results)
        resultss.append(results)

        with open(save_path / "params.json", "w") as f:
            json.dump(resultss, f)


if __name__ == "__main__":
    generated_data_path = sys.argv[1]
    training_data_path = sys.argv[2]
    stay_point_data_path = sys.argv[3]
    save_path = sys.argv[4]
    run(generated_data_path, training_data_path, stay_point_data_path, save_path)