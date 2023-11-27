import unittest
# use /data/geolife/100/narrow_200_10_bin30_seed0/training_data.csv as test data

import sys
import pandas as pd
import json
import folium
import matplotlib.pyplot as plt

sys.path.append("../priv_traj_gen")
from my_utils import load, save

sys.path.append("./")
import clustering
import make_traj_distribution
import competitors.clustering.clustering_generator as clustering_generator

class TestGenerator(unittest.TestCase):
    def setUp(self):
        # load distribution and id_to_traj
        with open("./test/data/noisy_traj_distribution.json", "r") as f:
            self.distribution = json.load(f)

        with open("./test/data/id_to_traj.json", "r") as f:
            self.id_to_traj = json.load(f)
            # convert str key to int key
            self.id_to_traj = {int(k):v for k,v in self.id_to_traj.items()}

        with open("./test/data/state_to_centroid_id.json", "r") as f:
            state_to_centroid_id = json.load(f)
            # convert str key to int key
            state_to_centroid_id = {int(k):int(v) for k,v in state_to_centroid_id.items()}

        self.generator = clustering_generator.Generator(self.distribution, self.id_to_traj, state_to_centroid_id)

    def test_seq_len_to_ids(self):
        ids = self.generator.seq_len_to_ids(3)

    def test_sample_from_ids(self):
        sampled_ids = self.generator.sample_from_ids([0,1,2])

    def test_make_sample(self):
        references = [[0,0], [0,0,0], [0,0,0,0]]
        sampled = self.generator.make_sample(references, None)
        for traj, reference in zip(sampled, references):
            self.assertEqual(len(traj), len(reference))
            self.assertEqual(traj[0], reference[0])

class TestClustering(unittest.TestCase):
    def setUp(self):
        # Load test data
        self.test_data = load("/data/geolife/100/narrow_200_10_bin30_seed0/training_data.csv")
        self.gps = pd.read_csv("/data/geolife/100/narrow_200_10_bin30_seed0/gps.csv", header=None).values
        self.k = 3

    def test_clustering(self):
        latlon_trajs = []

        for traj in self.test_data:
            latlon_trajs.append([tuple(self.gps[state]) for state in traj])

        id_to_centroid = clustering.clustering(latlon_trajs, self.k)

        with open("./test/data/id_to_centroid.json", "w") as f:
            json.dump(id_to_centroid, f)
            

    def test_make_state_to_centroid(self):

        with open("./test/data/id_to_centroid.json", "r") as f:
            id_to_centroid = json.load(f)
            # convert str key to int key
            id_to_centroid = {int(k):v for k,v in id_to_centroid.items()}

        state_to_centroid_id = clustering.make_state_to_centroid_id(self.gps, id_to_centroid)

        # save
        with open("./test/data/state_to_centroid_id.json", "w") as f:
            json.dump(state_to_centroid_id, f)

        # show the clustered states in the map with different colors
        
        m = folium.Map(location=[39.9, 116.3], zoom_start=12)
        for i in range(len(self.gps)):
            folium.CircleMarker(self.gps[i], radius=5, color="red" if state_to_centroid_id[i] == 0 else "blue" if state_to_centroid_id[i] == 1 else "green").add_to(m)
        # add markers for centroids
        for centroid in id_to_centroid.values():
            folium.Marker(location=centroid, icon=folium.Icon(color='black')).add_to(m)
            
        m.save("./test/data/map.html")

    def test_state_traj_to_centroid_id_traj(self):
        with open("./test/data/state_to_centroid_id.json", "r") as f:
            state_to_centroid_id = json.load(f)
            # convert str to int
            state_to_centroid_id = {int(k):int(v) for k,v in state_to_centroid_id.items()}

        for traj in self.test_data:
            centroid_id_traj = clustering.state_traj_to_centroid_id_traj(traj, state_to_centroid_id)

        with open("./test/data/id_to_centroid.json", "r") as f:
            id_to_centroid = json.load(f)
            # convert str key to int key
            id_to_centroid = {int(k):v for k,v in id_to_centroid.items()}

        # plot the state traj and centroid id traj in the map
        m = folium.Map(location=[39.9, 116.3], zoom_start=12)
        for i, state in enumerate(self.test_data[-1]):
            # with anotation of i
            folium.Marker(self.gps[state], popup=str(i)).add_to(m)

        for centroid_id in centroid_id_traj:
            folium.Marker(location=id_to_centroid[centroid_id], icon=folium.Icon(color='black')).add_to(m)

        m.save("./test/data/map_traj_to_centroid_id_traj.html")


    def test_run(self):
        centroid_trajs, state_to_centroid_id = clustering.run(self.test_data, self.gps, self.k)

        # save
        save("./test/data/centroid_trajs.csv", centroid_trajs)

        self.assertEqual(len(centroid_trajs), len(self.test_data))
        for centroid_traj, traj in zip(centroid_trajs, self.test_data):
            self.assertEqual(len(centroid_traj), len(traj))


class TestMakeTrajDistribution(unittest.TestCase):
    def setUp(self):
        self.test_data = load("./test/data/centroid_trajs.csv")
    
    def test_make_id_to_traj(self):
        id_to_traj = make_traj_distribution.make_id_to_traj(self.test_data)
        
        # save
        with open("./test/data/id_to_traj.json", "w") as f:
            json.dump(id_to_traj, f)

    def test_make_traj_distribution(self):
        with open("./test/data/id_to_traj.json", "r") as f:
            id_to_traj = json.load(f)
            # convert str key to int key
            id_to_traj = {int(k):v for k,v in id_to_traj.items()}

        counter = make_traj_distribution.make_traj_count(self.test_data)

        counts = [0]*len(id_to_traj)
        for i in range(len(id_to_traj)):
            counts[i] = counter[tuple(id_to_traj[i])]

        # plot the distribution, x_axis is traj_id
        plt.bar(range(len(counts)), counts)
        plt.savefig("./test/data/traj_distribution.png")
        plt.clf()

    def test_run(self):
        id_to_traj, noisy_traj_distribution = make_traj_distribution.run(self.test_data, 0)

        # save noisy_traj_distribution
        with open("./test/data/noisy_traj_distribution.json", "w") as f:
            json.dump(noisy_traj_distribution, f)

        plt.bar(range(len(noisy_traj_distribution)), noisy_traj_distribution)
        plt.savefig("./test/data/noisy_traj_distribution.png")
        plt.clf()


if __name__ == '__main__':
    unittest.main()