import pytest

import numpy as np
import torch


import sys
sys.path.append('./')
from dataset import TrajectoryDataset
from models import construct_generator, compute_loss_generator
from main import train_with_discrete_time

class TestGenarator:
    
    def setup_method(self, method):
        print("In method:", method)
        n_locations = 64
        self.hidden_dim = 30
        batch_size = 10
        n_data = 1000
        location_embedding_dim = 10
        n_split = 5
        time_dim = n_split+3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pad = np.ones((n_data, 1), dtype=int)
        traj1 = np.concatenate([np.zeros((n_data, 1)), pad, np.zeros((n_data, 1))], axis=1).tolist()
        traj2 = np.concatenate([np.ones((n_data, 1)), pad, 2*np.ones((n_data, 1))], axis=1).tolist()
        traj3 = np.concatenate([np.ones((n_data, 1))*2, pad, 30*np.ones((n_data, 1))], axis=1).tolist()
        traj = np.concatenate([traj1, traj2, traj3], axis=0).tolist()
        traj_time = [[0, 1, 2]]*n_data*3
        self.dataset = TrajectoryDataset(traj, traj_time, n_locations, n_split)
        traj_type_dim = len(self.dataset.label_to_reference)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=batch_size, collate_fn=self.dataset.make_padded_collate(True))

    def teardown_method(self, method):
        print("Out method:", method)


    @pytest.mark.parametrize(["model_name", "multitask", "consistent"], [("baseline", False, False), ("hrnet", False, False), ("hrnet", True, False), ("hrnet", True, True)])
    def test_make_samples(self, model_name, multitask, consistent):
        model = construct_generator(model_name, self.dataset.n_locations, self.dataset.n_time_split+1, self.hidden_dim, self.hidden_dim, self.hidden_dim, multitask, consistent)
        model = model.to(self.device)

        references = [(0,1,0), (13,1,2,3)]
        time_references = [0,0]
        trajs, _ = model.make_sample(references, time_references, 2)

        # Check if the first location is the same as the reference
        assert trajs[0][0] == 0
        assert trajs[1][0] == 13
        # Check if the length of the generated trajectory is the same as the reference
        assert len(trajs[0]) == len(references[0])
        assert len(trajs[1]) == len(references[1])

    

    @pytest.mark.parametrize(["model_name", "multitask", "consistent"], [("baseline", False, False), ("hrnet", False, False), ("hrnet", True, False), ("hrnet", True, True)])
    def test_train(self, model_name, multitask, consistent):
        model = construct_generator(model_name, self.dataset.n_locations, self.dataset.n_time_split+1, self.hidden_dim, self.hidden_dim, self.hidden_dim, multitask, consistent)
        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(1):
            for i, batch in enumerate(self.data_loader):
                input_locations = batch["input"].to(self.device)
                target_locations = batch["target"].to(self.device)
                references = [tuple(v) for v in batch["reference"]]
                input_times = batch["time"].to(self.device)
                target_times = batch["time_target"].to(self.device)
                
                _ = train_with_discrete_time(model, optimizer, compute_loss_generator, input_locations, target_locations, input_times, target_times, references, 1, 1, multitask)

        references = [(0,1,2), (1,1,2), (2,1,2)]
        time_references = [0,0,0]
        sampled, _ = model.make_sample(references, time_references, 3)

        expected = [[0,1,0], [1,1,2], [2,1,30]]
        assert sampled == expected

    # def test_embedding_position(self):
    #     model = construct_generator("hrnet", self.dataset.n_locations, self.dataset.n_time_split+1, self.hidden_dim, self.hidden_dim, self.hidden_dim, False, False)
    #     embedding_matrix = model.location_encoding_component.make_embedding_matrix(1, "cpu")[0]

    #     location_embedding = model.location_encoding_component(torch.tensor([0,0]).to("cpu"), depth=1)
    #     print(location_embedding)
    #     print(embedding_matrix[1])
    #     print(embedding_matrix[1] == location_embedding[0])
    #     print(location_embedding[0] == location_embedding[1])
    #     assert (embedding_matrix[1] == location_embedding[0]).all()
    #     # assert embedding_matrix[1] == location_embedding[1].detach().numpy()

    #     # location_embedding = model.location_encoding_component(torch.tensor(range(64)).to("cpu"), depth=2)
    #     # location_embedding = model.location_encoding_component(torch.tensor(range(64)).to("cpu"), depth=3)
    #     # print(location_embedding)