import unittest
from collections import Counter
# add parent path
import sys
import numpy as np
import pathlib
import torch
import folium
import sqlite3
import struct
import json

# from mocks import GeneratorMock, NameSpace
sys.path.append('./')
from evaluation import evaluate_next_location_on_test_dataset, count_passing_locations, compute_divergence
from my_utils import set_logger, load, plot_density, noise_normalize
from dataset import TrajectoryDataset
import evaluation
from grid import Grid
from run import construct_dataset


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
    args.compensation = True
    return args
    


class CompensateTrajsTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_compensate_trajs(self):
        dataset = "chengdu"
        n_bins = 30

        latlon_config_path = f"./config.json"
        with open(latlon_config_path, "r") as f:
            config = json.load(f)["latlon"][dataset]
        lat_range = config["lat_range"]
        lon_range = config["lon_range"]
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        grid = Grid(ranges)

        training_data_path = "/data/chengdu/10000/200_30_bin30_seed0/training_data.csv"
        training_data = load(training_data_path)
        route_training_data_path = "/data/chengdu/10000/200_30_bin30_seed0/route_training_data.csv"
        route_training_data = load(route_training_data_path)

        truncation = 23
        route_db_path = f"/data/{dataset}/pair_to_route/{n_bins}_tr{truncation}/paths.db"

        compensated, ids = evaluation.compensate_trajs(training_data, route_db_path)
        print(len(compensated)-len(ids))

        target = 14
        stay_traj = training_data[ids[target]]
        route_traj = route_training_data[ids[target]]

        latlons = [grid.state_to_center_latlon(state) for state in compensated[target]]
        route_latlons = [grid.state_to_center_latlon(state) for state in route_traj]
        m = folium.Map(location=[latlons[0][0], latlons[0][1]], zoom_start=13)
        # line
        folium.PolyLine(locations=latlons, color='red').add_to(m)
        folium.PolyLine(locations=route_latlons, color='blue').add_to(m)
        # save by png
        m.save(f"./test/data/compensated_traj.html")


        # with sqlite3.connect(route_db_path) as conn:
        #     # count the number of row in state_edge_to_route
        #     cursor = conn.cursor()
        #     cursor.execute("SELECT * FROM state_edge_to_route WHERE start_state=203")
        #     n_rows = cursor.fetchall()
        #     # for line in n_rows:
        #         # print(eval(line[2]))


        # trajs = load(f"/data/{dataset}/100/200_10_bin14_seed0/training_data.csv")
        # print(trajs)
        # compensated_trajs = evaluation.compensate_trajs(trajs, route_db_path)
        # print(compensated_trajs)
        # # self.assertEqual(len(compensated_trajs), len(trajs))
        # # for traj, compensated_traj in zip(trajs, compensated_trajs):
        #     # self.assertEqual(traj[0], compensated_traj[0])
        #     # self.assertEqual(traj[-1], compensated_traj[-1])
        
        # latlon_traj = [grid.state_to_center_latlon(state) for state in compensated_trajs[0]]
        # print(latlon_traj)
        # # plot by folium with anotation
        # m = folium.Map(location=[latlon_traj[0][0], latlon_traj[0][1]], zoom_start=13)
        # for latlon, state in zip(latlon_traj, compensated_trajs[0]):
        #     m.add_child(folium.Marker(location=latlon, icon=folium.Icon(color='red'), popup=str(state)))
        # m.save(f"./test/data/compensated_traj.html")


class EvaluationMTNetTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_mock(self):
        # orig_counts = count_passing_locations(self.original_data)
        # gene_counts = count_passing_locations(self.generated_data)
        # plot_density(orig_counts, self.n_locations, "./test/orig.png")
        # plot_density(gene_counts, self.n_locations, "./test/gene.png")

        # divergence = compute_divergence(orig_counts, len(self.original_data), gene_counts, len(self.generated_data), 32*32, positive=True)
        # print("positive", divergence)
        # divergence = compute_divergence(orig_counts, len(self.original_data), gene_counts, len(self.generated_data), 32*32, positive=False)
        # print("kl", divergence)
        # divergence = compute_divergence(orig_counts, sum(orig_counts.values()), gene_counts, sum(gene_counts.values()), 32*32, axis=1)
        # print("js", divergence)
        dataset = "chengdu"
        traj_path = f"/data/results/chengdu/10000/MTNet/samples_1.txt"
        time_traj_path = f"/data/results/chengdu/10000/MTNet/samples_t_1.txt"
        n_bins = 30

        data_path = pathlib.Path("/data/chengdu/10000/200_10_bin30_seed0/training_data.csv").parent
        route_data_path = pathlib.Path("/data/chengdu/10000/0_0_bin30_seed0/training_data.csv").parent

        logger = set_logger(__name__, "./test/test.log")

        generator = evaluation.MTNetGeneratorMock(traj_path, time_traj_path, dataset, n_bins)
        dataset = construct_dataset(data_path, route_data_path, 5, "chengdu")
        evaluation.compute_auxiliary_information(dataset, "./test/data/a", logger)
        args = evaluation.set_args()
        args.route_generator = True
        args.save_dir = "./test/data"
        results = evaluation.run(generator, dataset, args)
        print(results)



class EvaluationPrivTraceTestCase(unittest.TestCase):

    def setUp(self):
        n_bins = 30
        generated_data_path = "/data/results/chengdu/10000/privtrace_seed0_eps0/evaluated_0.csv"
        stay_point_data_path = "/data/chengdu/10000/200_60_bin30_seed0/training_data.csv"
        route_data_path = "/data/chengdu/10000/0_0_bin30_seed0/training_data.csv"

        self.stay_point_data = load(stay_point_data_path)
        self.generated_data = load(generated_data_path)
        self.route_data = load(route_data_path)

        self.n_locations = (n_bins+2)**2


        self.base_location_counts = Counter([row[0] for row in self.stay_point_data if len(row) > 1])
        self.top_base_locations = [location for location, _ in self.base_location_counts.most_common(30)]
        self.route_base_location_counts = Counter([row[0] for row in self.route_data if len(row) > 1])

        # print(len(self.stay_point_data), len(self.route_data))
        # print(len([row[0] for row in self.stay_point_data if len(row) > 1]), len([row[0] for row in self.route_data if len(row) > 1]))
        # print(self.base_location_counts)
        # print(self.route_base_location_counts)
        # print(sum(self.base_location_counts.values()), sum(self.route_base_location_counts.values()))
        # print(len(self.base_location_counts), len(self.route_base_location_counts))
        # print(self.base_location_counts == self.route_base_location_counts)


    def test_route_evaluation(self):
        route_db_path = "/data/chengdu/pair_to_route/narrow_30/paths.db"
        compensated_trajs = evaluation.compensate_trajs(self.generated_data, route_db_path)

        first_location_counts = Counter([row[0] for row in compensated_trajs])

        for location in self.top_base_locations:
            print(location)
            orig_counts = evaluation.count_route_locations(self.route_data, location)
            gene_counts = evaluation.count_route_locations(compensated_trajs, location)

            plot_density(orig_counts, self.n_locations, f"./test/imgs/orig_{location}.png")
            plot_density(gene_counts, self.n_locations, f"./test/imgs/gene_{location}.png")

            # print("route", self.route_base_location_counts[location], "stay", self.base_location_counts[location])

            divergence = compute_divergence(orig_counts, self.route_base_location_counts[location], gene_counts, first_location_counts[location], self.n_locations, positive=True)
            assertNotEqual(divergence, float("inf"))
            # print("positive", divergence)
            divergence = compute_divergence(orig_counts, self.route_base_location_counts[location], gene_counts, first_location_counts[location], self.n_locations, positive=False)
            assertNotEqual(divergence, float("inf"))
            # print("kl", divergence)
            divergence = compute_divergence(orig_counts, sum(orig_counts.values()), gene_counts, sum(gene_counts.values()), self.n_locations, axis=1)
            assertNotEqual(divergence, float("inf"))
            # print("js", divergence)

class EvaluationHieMRNetTestCase(unittest.TestCase):

    def setUp(self):
        print("method", self._testMethodName)
        target_epoch = 60
        n_bins = 30
        generated_data_path = f"/data/results/chengdu/10000/200_60_bin30_seed0/fulllinear_quadtree_dpTrue_meta10000_dim100_64_64_256_btch0_cldepth_1000_trTrue_coTrue/evaluated_{target_epoch}.csv"
        stay_point_data_path = "/data/chengdu/10000/200_60_bin30_seed0/training_data.csv"
        route_data_path = "/data/chengdu/10000/0_0_bin30_seed0/training_data.csv"

        self.stay_point_data = load(stay_point_data_path)
        self.generated_data = load(generated_data_path)
        self.route_data = load(route_data_path)

        self.n_locations = (n_bins+2)**2


        self.base_location_counts = Counter([row[0] for row in self.stay_point_data if len(row) > 1])
        self.top_base_locations = [location for location, _ in self.base_location_counts.most_common(30)]
        self.route_base_location_counts = Counter([row[0] for row in self.route_data if len(row) > 1])
        

    def test_route_evaluation(self):
        route_db_path = "/data/chengdu/pair_to_route/30/paths.db"
        compensated_trajs = evaluation.compensate_trajs(self.generated_data, route_db_path)

        route_first_location_counts = Counter([row[0] for row in compensated_trajs if len(row) > 1])
        first_location_counts = Counter([row[0] for row in self.generated_data if len(row) > 1])
        # print(len(first_location_counts_a), len(first_location_counts_b))
        # print(first_location_counts_a[801], first_location_counts_b[801])
        # for traj_a, traj_b in zip([row for row in compensated_trajs], [row for row in self.generated_data]):
        #     assert traj_a[0] == traj_b[0]
        #     assert traj_a[-1] == traj_b[-1]

        avg = 0

        for location in self.top_base_locations:
            print(location)
            orig_counts = evaluation.count_route_locations(self.route_data, location)
            gene_counts = evaluation.count_route_locations(compensated_trajs, location)

            plot_density(orig_counts, self.n_locations, f"./test/imgs/orig_{location}.png")
            plot_density(gene_counts, self.n_locations, f"./test/imgs/gene_{location}.png")

            print("orig", self.route_base_location_counts[location], "gene", route_first_location_counts[location])
            print(orig_counts)
            print(gene_counts)

            divergence = compute_divergence(orig_counts, self.route_base_location_counts[location], gene_counts, route_first_location_counts[location], self.n_locations, positive=True)
            self.assertNotEqual(divergence, float("inf"))
            print("positive", divergence)
            avg += divergence / 30
            divergence = compute_divergence(orig_counts, self.route_base_location_counts[location], gene_counts, route_first_location_counts[location], self.n_locations, positive=False)
            self.assertNotEqual(divergence, float("inf"))
            print("kl", divergence)
            divergence = compute_divergence(orig_counts, sum(orig_counts.values()), gene_counts, sum(gene_counts.values()), self.n_locations, axis=1)
            self.assertNotEqual(divergence, float("inf"))
            print("js", divergence)
        
        print(avg)


    def test_destination_evaluation(self):
        route_db_path = "/data/chengdu/pair_to_route/30/paths.db"
        compensated_trajs = evaluation.compensate_trajs(self.generated_data, route_db_path)

        route_first_location_counts = Counter([row[0] for row in compensated_trajs if len(row) > 1])
        first_location_counts = Counter([row[0] for row in self.generated_data if len(row) > 1])

        for location in self.top_base_locations:
            print(location)
            orig_counts = evaluation.count_route_locations(self.stay_point_data, location)
            gene_counts = evaluation.count_route_locations(self.generated_data, location)

            plot_density(orig_counts, self.n_locations, f"./test/imgs/orig_{location}.png")
            plot_density(gene_counts, self.n_locations, f"./test/imgs/gene_{location}.png")

            print("route", route_first_location_counts[location], "stay", first_location_counts[location])

            divergence = compute_divergence(orig_counts, self.base_location_counts[location], gene_counts, first_location_counts[location], self.n_locations, positive=True)
            self.assertNotEqual(divergence, float("inf"))
            print("positive", divergence)
            divergence = compute_divergence(orig_counts, self.base_location_counts[location], gene_counts, first_location_counts[location], self.n_locations, positive=False)
            self.assertNotEqual(divergence, float("inf"))
            print("kl", divergence)
            divergence = compute_divergence(orig_counts, sum(orig_counts.values()), gene_counts, sum(gene_counts.values()), self.n_locations, axis=1)
            self.assertNotEqual(divergence, float("inf"))
            print("js", divergence)


class EvaluationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        print(self._testMethodName)
        self.n_locations = 16

        traj1 = [0,1,2,3]
        traj2 = [0,1]
        traj3 = [1,2]
        self.route_trajs = [traj1, traj2, traj3]
        self.stay_point_trajs = [[traj1[0], traj1[-1]], [traj2[0], traj2[-1]], [traj3[0], traj3[-1]]]

        self.base_location_counts = Counter([traj[0] for traj in self.stay_point_trajs if len(traj) > 1])
        self.route_base_location_counts = Counter([traj[0] for traj in self.route_trajs if len(traj) > 1])

        self.top_base_locations = [location for location, _ in self.base_location_counts.most_common(30)]
        self.top_route_base_locations = [location for location, _ in self.route_base_location_counts.most_common(30)]

    def test_make_downsample_dict(self):
        to_bin = 14
        from_bin = 30
        downsample_dict = evaluation.make_downsampling_dict(from_bin=from_bin, to_bin=to_bin)
        from collections import Counter

        trajs_14 = load("/data/geolife_mm/0/200_30_bin14_seed0/training_data.csv")
        trajs_30 = load("/data/geolife_mm/0/200_30_bin30_seed0/training_data.csv")

        downsampled_trajs, indice = evaluation.downsample_trajs(trajs_30, downsample_dict)
        print(len(trajs_30)-len(indice), "trajectories removed")

        print(downsampled_trajs)

        # for traj1, traj2 in zip(trajs_14, downsampled_trajs):
            # print("orig", traj1)
            # print("down", traj2)

    def test_compensate_trajs(self):
        dataset = "chengdu"
        n_bins = 14

        latlon_config_path = f"./config.json"
        with open(latlon_config_path, "r") as f:
            config = json.load(f)["latlon"][dataset]
        lat_range = config["lat_range"]
        lon_range = config["lon_range"]
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        grid = Grid(ranges)

        route_db_path = f"/data/{dataset}/pair_to_route/{n_bins}/paths.db"

        with sqlite3.connect(route_db_path) as conn:
            # count the number of row in state_edge_to_route
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM state_edge_to_route WHERE start_state=203")
            n_rows = cursor.fetchall()
            # for line in n_rows:
                # print(eval(line[2]))


        trajs = load(f"/data/{dataset}/100/200_10_bin14_seed0/training_data.csv")
        print(trajs)
        compensated_trajs = evaluation.compensate_trajs(trajs, route_db_path)
        print(compensated_trajs)
        # self.assertEqual(len(compensated_trajs), len(trajs))
        # for traj, compensated_traj in zip(trajs, compensated_trajs):
            # self.assertEqual(traj[0], compensated_traj[0])
            # self.assertEqual(traj[-1], compensated_traj[-1])
        
        latlon_traj = [grid.state_to_center_latlon(state) for state in compensated_trajs[0]]
        print(latlon_traj)
        # plot by folium with anotation
        m = folium.Map(location=[latlon_traj[0][0], latlon_traj[0][1]], zoom_start=13)
        for latlon, state in zip(latlon_traj, compensated_trajs[0]):
            m.add_child(folium.Marker(location=latlon, icon=folium.Icon(color='red'), popup=str(state)))
        m.save(f"./test/data/compensated_traj.html")
        

    def test_route_evaluation(self):

        compensated_trajs = [[0,1,2,3,4], [0,1], [1,2]]
        first_location_counts = Counter([row[0] for row in compensated_trajs if len(row) > 1])

        for location in self.top_base_locations:
            print(location)
            orig_counts = evaluation.count_route_locations(self.route_trajs, location)
            gene_counts = evaluation.count_route_locations(compensated_trajs, location)

            plot_density(orig_counts, self.n_locations, f"./test/imgs/orig_{location}.png")
            plot_density(gene_counts, self.n_locations, f"./test/imgs/gene_{location}.png")

            # print("route", self.route_base_location_counts[location], "stay", self.base_location_counts[location])

            divergence = compute_divergence(orig_counts, self.route_base_location_counts[location], gene_counts, first_location_counts[location], self.n_locations, positive=True)
            self.assertNotEqual(divergence, float("inf"))
            print("positive", divergence)
            divergence = compute_divergence(orig_counts, self.route_base_location_counts[location], gene_counts, first_location_counts[location], self.n_locations, positive=False)
            self.assertNotEqual(divergence, float("inf"))
            print("kl", divergence)
            divergence = compute_divergence(orig_counts, sum(orig_counts.values()), gene_counts, sum(gene_counts.values()), self.n_locations, axis=1)
            self.assertNotEqual(divergence, float("inf"))
            print("js", divergence)

    def test_target_evaluation(self):

        compensated_trajs = [[1,0], [1,0], [1,0], [0], [0,1], [0]]
        first_location_counts = Counter([row[0] for row in compensated_trajs if len(row) > 1])

        for location in self.top_base_locations:
            print(location)
            orig_counts = evaluation.count_target_locations(self.route_trajs, location)
            gene_counts = evaluation.count_target_locations(compensated_trajs, location)

            print(gene_counts)

            plot_density(orig_counts, self.n_locations, f"./test/imgs/orig_{location}.png")
            plot_density(gene_counts, self.n_locations, f"./test/imgs/gene_{location}.png")

            # print("route", self.route_base_location_counts[location], "stay", self.base_location_counts[location])

            divergence = compute_divergence(orig_counts, self.route_base_location_counts[location], gene_counts, first_location_counts[location], self.n_locations, positive=True)
            self.assertNotEqual(divergence, float("inf"))
            print("positive", divergence)
            divergence = compute_divergence(orig_counts, self.route_base_location_counts[location], gene_counts, first_location_counts[location], self.n_locations, positive=False)
            self.assertNotEqual(divergence, float("inf"))
            print("kl", divergence)
            divergence = compute_divergence(orig_counts, sum(orig_counts.values()), gene_counts, sum(gene_counts.values()), self.n_locations, axis=1)
            self.assertNotEqual(divergence, float("inf"))
            print("js", divergence)

    def test_passing_evaluation(self):

        compensated_trajs = [[0,1,2,1,3,1,4,1], [0,1], [1,2]]
        first_location_counts = Counter([row[0] for row in compensated_trajs if len(row) > 1])


        orig_counts = evaluation.count_passing_locations(self.route_trajs)
        gene_counts = evaluation.count_passing_locations(compensated_trajs)

        plot_density(orig_counts, self.n_locations, f"./test/imgs/orig.png")
        plot_density(gene_counts, self.n_locations, f"./test/imgs/gene.png")

        # print("route", self.route_base_location_counts[location], "stay", self.base_location_counts[location])

        divergence = compute_divergence(orig_counts, len(self.stay_point_trajs), gene_counts, len(compensated_trajs), self.n_locations, positive=True)
        self.assertNotEqual(divergence, float("inf"))
        print("positive", divergence)
        divergence = compute_divergence(orig_counts, len(self.stay_point_trajs), gene_counts, len(compensated_trajs), self.n_locations, positive=False)
        self.assertNotEqual(divergence, float("inf"))
        print("kl", divergence)
        divergence = compute_divergence(orig_counts, sum(orig_counts.values()), gene_counts, sum(gene_counts.values()), self.n_locations, axis=1)
        self.assertNotEqual(divergence, float("inf"))
        print("js", divergence)

    def test_compute_divergence(self):
        count1 = Counter([2,3,4,3,4,4])
        count2 = Counter([0,1,2,3,4,3,4,4])
        divergence = compute_divergence(count1, 3, count2, 3, 5, positive=True)
        self.assertAlmostEqual(divergence, 0.0) # due to the positive flag
        divergence = compute_divergence(count1, 3, count2, 3, 5, positive=False)
        print(divergence, "kl")
        self.assertNotEqual(divergence, 0.0) # due to the positive flag
        divergence = compute_divergence(count1, 3, count2, 3, 5, positive=False, kl=False)
        print(divergence, "a")

    # def test_evaluate_next_location_on_test_dataset(self):

    #     n_locations = 16
    #     n_data = 120
    #     n_split = 7
    #     n_kinds = 3
    #     n_test_locations = 2
    #     traj1 = np.concatenate([np.zeros((int(n_data/n_kinds), 1)).astype(int), np.ones((int(n_data/n_kinds), 1)).astype(int), 2*np.ones((int(n_data/n_kinds), 1)).astype(int)], axis=1).tolist()
    #     traj2 = np.concatenate([np.zeros((int(n_data/n_kinds), 1)).astype(int), np.ones((int(n_data/n_kinds), 1)).astype(int), 3*np.ones((int(n_data/n_kinds), 1)).astype(int)], axis=1).tolist()
    #     traj3 = np.concatenate([4*np.ones((int(n_data/n_kinds), 1)).astype(int), np.ones((int(n_data/n_kinds), 1)).astype(int), 3*np.ones((int(n_data/n_kinds), 1)).astype(int)], axis=1).tolist()
    #     traj = np.concatenate([traj1, traj2, traj3], axis=0).tolist()
    #     traj_time = [[0, 1, 2]]*n_data
    #     dataset = TrajectoryDataset(traj, traj_time, n_locations, n_split)
    #     dataset.compute_auxiliary_information(self.save_path, self.logger)
    #     first_order_test_data_loader, first_counters = dataset.make_first_order_test_data_loader(n_test_locations)
    #     second_order_test_data_loader, second_counters = dataset.make_second_order_test_data_loader(n_test_locations)
    #     first_next_location_counts = dataset.first_next_location_counts
    #     second_order_next_location_counts = dataset.second_order_next_location_counts

    #     time_output = None
    #     def location_output(input):
    #         output_locations = []
    #         for traj in input:
    #             first_location = traj[1].item()
    #             second_location = traj[2].item()
    #             probs = []
    #             probs.append([0]*n_locations)
    #             probs.append(noise_normalize(first_next_location_counts[first_location]))
    #             probs.append(noise_normalize(second_order_next_location_counts[(first_location, second_location)]))
    #             probs = torch.tensor(probs)
    #             output_locations.append(probs)
    #         output_locations = torch.stack(output_locations)
    #         return torch.log(output_locations)
    #     generator_mock = GeneratorMock(location_output, time_output)

    #     first_jss = evaluation.evaluate_next_location_on_test_dataset(first_next_location_counts, first_order_test_data_loader, first_counters, generator_mock, 1)
    #     second_jss = evaluation.evaluate_next_location_on_test_dataset(second_order_next_location_counts, second_order_test_data_loader, second_counters, generator_mock, 2)

    #     self.assertEqual(first_jss, [[[0,0]], [[0,0]]])
    #     self.assertEqual(second_jss, [[[0,0]], [[0,0]]])

    # # def test_count_passing(self):
    # #     # print(self.dataset.data)
    # #     a = count_passing_locations(self.dataset.data)
    # #     print(a)

    # def test_run(self):

    #     save_path = pathlib.Path("./test/test/test/test_run.log")
    #     logger = set_logger(__name__, save_path)
    #     distance_matrix = torch.zeros(16, 16)
    #     np.save("/data/test_run.log/distance_matrix_bin2.npy", distance_matrix)
    #     next_location_distributions_ = torch.zeros(3, 16)
    #     next_location_distributions_[0, 1] = 0.5
    #     next_location_distributions_[0, 2] = 0.5
    #     next_location_distributions_[1, 1] = 1
    #     next_location_distributions_[2, 2] = 1
    #     time_output = None
    #     # seq_len = 3
    #     def location_output(input):
    #         return torch.log(next_location_distributions_.repeat(1,3)).view(-1, 3, 16)
    #     generator = GeneratorMock(location_output, time_output)
    #     args = make_args()
    #     epoch = 1
    #     dataset = make_data()
    #     dataset.compute_auxiliary_information(save_path, logger)
    #     dataset.make_first_order_test_data_loader(5)
    #     dataset.make_second_order_test_data_loader(5)
    #     results = evaluation.run(generator, dataset, args, epoch)
    #     print(results)


    # def test_compensate_edge_by_map(self):

    #     def plot_latlons_on_map(latlons, color='red'):
    #         m = folium.Map(location=[latlons[0][0], latlons[0][1]], zoom_start=13)
    #         folium.Marker(location=latlons[0], icon=folium.Icon(color='green')).add_to(m)
    #         folium.Marker(location=latlons[-1], icon=folium.Icon(color='red')).add_to(m)
    #         for i in range(len(latlons)-1):
    #             folium.PolyLine(locations=[latlons[i], latlons[i+1]], color=color).add_to(m)
    #         return m
    #     n_bins = 30
    #     lat_range = [39.85, 40.1]
    #     lon_range = [116.25, 116.5]
    #     ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    #     grid = Grid(ranges)

    #     from_state = 500
    #     to_state = 10

    #     db_path = f"/data/geolife/pair_to_route/narrow_{n_bins}/paths.db"
    #     compensated_states = evaluation.compensate_edge_by_map(from_state, to_state, db_path)
    #     latlons = [grid.state_to_center_latlon(state) for state in compensated_states]
    #     m = plot_latlons_on_map(latlons)
    #     # save by png
    #     m.save(f"./test/compensated_edge.html")

    #     reversed_compensated_states = evaluation.compensate_edge_by_map(to_state, from_state, db_path)
    #     self.assertEqual(reversed_compensated_states, compensated_states[::-1])

if __name__ == "__main__":
    unittest.main()