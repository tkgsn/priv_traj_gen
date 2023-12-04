import sys
import pathlib
import pandas as pd
import json
from logging import getLogger

import clustering
from clustering_generator import ClusteringGenerator
import make_traj_distribution
import matplotlib.pyplot as plt
import pickle
import argparse

sys.path.append("../../")
from my_utils import load
# from dataset import TrajectoryDataset
# import evaluation

# def set_args():

#     class Namespace():
#         pass

#     args = Namespace()
#     args.evaluate_global = False
#     args.evaluate_passing = True
#     args.evaluate_source = True
#     args.evaluate_target = True
#     args.evaluate_route = True
#     args.evaluate_destination = True
#     args.evaluate_distance = True
#     args.evaluate_first_next_location = False
#     args.evaluate_second_next_location = False
#     args.evaluate_second_order_next_location = False
#     args.eval_initial = True
#     args.eval_interval = 1
#     args.compensation = True
#     args.n_test_locations = 30
#     args.dataset = "geolife"
#     args.n_split = 5
#     # this is not used
#     args.batch_size = 100

#     return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--training_data_name', type=str)
    parser.add_argument('--k', type=int)
    parser.add_argument('--epsilon', type=float)
    args = parser.parse_args()
    args.network_type = "clustering"

    training_data_dir = pathlib.Path(f"/data/{args.dataset}/{args.data_name}/{args.training_data_name}")
    save_dir = pathlib.Path(f"/data/{args.dataset}/{args.data_name}/{args.training_data_name}/clustering_{args.k}")
    k = args.k
    epsilon = args.epsilon

    save_dir = save_dir / "model_{}".format(epsilon)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "imgs").mkdir(parents=True, exist_ok=True)

    print("load data from", training_data_dir)
    training_data = load(training_data_dir / "training_data.csv")
    time_data = load(training_data_dir / "training_data_time.csv")
    # route_data = load(training_data_dir / "route_training_data.csv")
    route_data = None
    gps = pd.read_csv(training_data_dir / "gps.csv", header=None).values

    print("clustering...")
    centroid_trajs, state_to_centroid_id = clustering.run(training_data, gps, k)

    print("making traj distribution...")
    id_to_traj, noisy_traj_distribution = make_traj_distribution.run(centroid_trajs, epsilon)

    with open(save_dir / "id_to_traj.json", "w") as f:
        json.dump(id_to_traj, f)
        print("save id_to_traj to", save_dir / "id_to_traj.json")
    plt.bar(range(len(noisy_traj_distribution)), noisy_traj_distribution)
    plt.savefig(save_dir / "imgs" / "noisy_traj_distribution.png")
    plt.clf()
    print("save noisy_traj_distribution to", save_dir / "imgs" / "noisy_traj_distribution.png")

    print("preparing generator...")
    generator = ClusteringGenerator(noisy_traj_distribution, id_to_traj, state_to_centroid_id)

    # save generator
    with open(save_dir / f"generator.pickle", "wb") as f:
        pickle.dump(generator, f)
        print("save generator to", save_dir / f"generator.pickle")

    with open(save_dir.parent / "params.json", "w") as f:
        json.dump(vars(args), f)
        print("save params to", save_dir.parent / "params.json")
    # args = set_args()
    # args.save_path = str(save_dir)

    # load setting file
    # with open(training_data_dir / "params.json", "r") as f:
    #     param = json.load(f)
    # n_bins = param["n_bins"]
    # n_locations = (n_bins+2)**2


    # for evaluation, we use stay_point_trajectories with route_data
    # dataset = TrajectoryDataset(training_data, time_data, n_locations, args.n_split, route_data=route_data)
    # dataset.compute_auxiliary_information(save_dir, getLogger(__name__))

    # results = [evaluation.run(generator, dataset, args, 0)]
    # print("save results to", save_dir / "params.json")
    # with open(save_dir / "params.json", "w") as f:
        # json.dump(results, f)