import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import torch
import numpy as np
from opacus.layers.dp_rnn import DPGRU, DPGRUCell
from dataset import TrajectoryDataset


class BaseGRUNet(nn.Module):

    def __init__(self, reference_to_label, input_dim, embed_size, traj_type_dim, hidden_dim, n_layers, time_dim):
        super(BaseGRUNet, self).__init__()
        self.traj_type_embedding = nn.Embedding(traj_type_dim, hidden_dim*n_layers)
        self.reference_to_label = reference_to_label
        self.location_embedding = nn.Embedding(input_dim, embed_size)
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.gru = DPGRUCell(embed_size+time_dim, hidden_dim, True)
        self.relu = nn.ReLU()

    def input_gru(self, x, references):
        state = self.init_hidden(references)
        locations = x[0]
        times = x[1]

        seq_len = locations.shape[1]

        states = []
        for i in range(seq_len):
            input_locations = locations[:, i]
            input_times = times[:, i]
            state = self.encode(input_locations, input_times, state)
            states.append(state)
        
        out = torch.stack(states, dim=1)

        return out
    
    def forward(self, x, references):
        # embedding prefix
        out = self.input_gru(x, references)

        # decoding to next location and next time
        location, time = self.decode(out)
        return location, time

    def init_hidden(self, references):
        labels = torch.tensor([self.reference_to_label[reference] for reference in references]).to(next(self.parameters()).device)
        hidden = self.traj_type_embedding(labels).view(-1, self.hidden_dim)
        return hidden

    def step(self, inputs, input_times, input_states, remove_locationss=None):
        x = self.location_embedding(inputs)
        # one hot encode time
        times = F.one_hot(input_times, num_classes=self.time_dim).long().view(input_times.shape[0], -1)
        # concat time information
        x = torch.cat([x, times], dim=-1)
        state = self.gru(x, input_states)

        location, time = self.decode(state, remove_locationss)
        return state, location, time
    

    def make_sample(self, references, n_locations, time_end_index, batch_size, real_start=None):
        # the indice of references should correspond to the indice of real_start
        n_samples = len(references)
        start_idx = TrajectoryDataset.start_idx(n_locations)
        device = next(self.parameters()).device

        trajectories = []
        time_trajectories = []
        for i in range(int(n_samples / batch_size)):
            sampled_trajectory = torch.tensor([start_idx for _ in range(batch_size)]).to(device, non_blocking=True).view(batch_size, 1)
            sampled_time_trajectory = torch.tensor([0 for _ in range(batch_size)]).to(device, non_blocking=True).view(batch_size, 1)
            input_references = references[i*batch_size:(i+1)*batch_size]
            states = [self.init_hidden(input_references)]
            seq_len = max([len(reference) for reference in input_references])
            batch_references = np.array([list(reference) + [i for i in range(len(reference), seq_len)] for reference in input_references])
            for j in range(seq_len):
                inputs = sampled_trajectory[:, -1]
                input_times = sampled_time_trajectory[:, -1]
                input_states = states[-1]
                output_states, pred_location, pred_time = self.step(inputs, input_times, input_states, sampled_trajectory[:,1:])
                if j == 0 and real_start is not None:
                    locations = real_start[0][i*batch_size:(i+1)*batch_size]
                    times = real_start[1][i*batch_size:(i+1)*batch_size]
                else:
                    locations = torch.exp(pred_location).multinomial(1).squeeze()
                    times = torch.exp(pred_time).multinomial(1).squeeze()

                sampled_trajectory = torch.cat([sampled_trajectory, locations.view(batch_size, 1)], dim=1)
                # access sampled_trajectory by batch_references[:, j]
                sampled_trajectory[:, -1] = sampled_trajectory[range(batch_size), batch_references[:, j]+1]
                sampled_time_trajectory = torch.cat([sampled_time_trajectory, times.view(batch_size, 1)], dim=1)
                # sampled_trajectory.append(locations)
                # sampled_time_trajectory.append(times)
                states.append(output_states)
                
            sampled_trajectory = sampled_trajectory[:,1:].cpu().detach().numpy().tolist()
            sampled_time_trajectory = sampled_time_trajectory[:,1:].cpu().detach().numpy().tolist()
            trajectories.extend(sampled_trajectory)
            time_trajectories.extend(sampled_time_trajectory)
        
        # remove the outside of the format
        for i in range(len(trajectories)):
            length = len(references[i])
            trajectories[i] = trajectories[i][:length]
            time_trajectories[i].append(time_end_index)
            time_trajectories[i] = time_trajectories[i][:length+1]    
        
        return trajectories, time_trajectories

    def encode(self, location, time, state):
        x = self.location_embedding(location)
        # one hot encode time
        times = F.one_hot(time, num_classes=self.time_dim).long().view(time.shape[0], -1)
        # concat time information
        x = torch.cat([x, times], dim=-1)

        state = self.gru(x, state)
        return state

    def decode(self, embedding, remove_locationss=None):
        pass     

    def remove_location(self, location, remove_locationss):
        # remove the locations that are in remove_locations
        if remove_locationss is not None:
            assert location.shape[0] == len(remove_locationss)
            for i, remove_locations in enumerate(remove_locationss):
                location[i, remove_locations] = -float("inf")
        return location


class GRUNet(BaseGRUNet):
    def __init__(self, input_dim, traj_type_dim, hidden_dim, output_dim, time_dim, n_layers, embed_size, reference_to_label):
        super(GRUNet, self).__init__(reference_to_label, input_dim, embed_size, traj_type_dim, hidden_dim, n_layers, time_dim)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        if self.n_layers >= 2:
            print("CONSTRUCTING")
        self.time_dim = time_dim
        self.output_dim = output_dim

        # output_dim + time_dim because we also want to predict the time
        self.fc = nn.Linear(hidden_dim, output_dim+time_dim)
        
    
    def decode(self, embedding, remove_locationss=None):
        out = self.fc(self.relu(embedding))
        # split the last dimension into location and time
        location, time = torch.split(out, [self.output_dim, self.time_dim], dim=-1)

        # remove the locations that are in remove_locations
        location = self.remove_location(location, remove_locationss)
        
        # convert to log_probs
        location = F.log_softmax(location, dim=-1)
        time = F.log_softmax(time, dim=-1)
        return location, time        


class MetaGRUNet(BaseGRUNet):
    def __init__(self, meta_network, input_dim, traj_type_dim, hidden_dim, output_dim, time_dim, n_layers, embed_size, reference_to_label):
        self.hidden_dim = hidden_dim+meta_network.n_classes+meta_network.n_margin
        super(MetaGRUNet, self).__init__(reference_to_label, input_dim, embed_size, traj_type_dim, self.hidden_dim, n_layers, time_dim)
        self.meta_net = meta_network

        self.fc1 = nn.Linear(self.hidden_dim, embed_size)
        self.fc2 = nn.Linear(embed_size, self.hidden_dim)
        self.fc_time = nn.Linear(self.hidden_dim, time_dim)
    
    def decode(self, embedding, remove_locationss=None):
        out = self.fc1(embedding)
        out = self.fc2(self.relu(out))
        # softmax
        location = self.meta_net(out)
        # remove the locations that are in remove_locations
        location = self.remove_location(location, remove_locationss)
        time = self.fc_time(out)
        time = F.log_softmax(time, dim=-1)
        return location, time


class MetaNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MetaNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.n_classes = input_dim

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(self.relu(out))
        return F.log_softmax(out, dim=-1)


class MetaAttentionNetwork(nn.Module):
    """
    This network is pre-trained using the probability distributions
    This intends to implicitly memory n_memories distributions by self.embeddings

    input: query := (batch_size * embed_dim)
    output: log_probs := (batch_size * n_locations)
    """
    
    def __init__(self, memory_dim, hidden_dim, n_locations, n_classes):
        super(MetaAttentionNetwork, self).__init__()
        self.n_classes = n_classes
        self.memory_dim = memory_dim
        self.n_margin = 0

        # prepare the trainable parameters with size n_memories * memory_dim * 2
        self.embeddings = nn.Parameter(torch.empty(memory_dim*2, n_classes+self.n_margin))
        self.embeddings_bias = nn.Parameter(torch.empty(memory_dim*2))
        self.init_embeddings()
        self.embeddings_query = nn.Linear(n_classes+self.n_margin, memory_dim)
        
        self.fc1 = nn.Linear(memory_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_locations)
        self.relu = nn.ReLU()

    def init_embeddings(self):
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.embeddings, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.embeddings)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.embeddings_bias, -bound, bound)

    def forward(self, query):
        # memory
        keys = self.embeddings.T[:, :self.memory_dim] + self.embeddings_bias[:self.memory_dim]
        values = self.embeddings.T[:, self.memory_dim:] + self.embeddings_bias[self.memory_dim:]

        if query.shape[-1] > self.n_classes+self.n_margin:
            query, input_embeddings = torch.split(query, [self.n_classes+self.n_margin, query.shape[-1]-self.n_classes-self.n_margin], dim=-1)
            query = F.softmax(query, dim=-1)
            if query.shape[0] > 1:
                # print(query[0][1])
                pass
        else:
            input_embeddings = torch.zeros(query.shape[0], self.memory_dim, device=query.device)
         
        query = self.embeddings_query(query)
        extracted_memory = attention(query, keys, values)
        extracted_memory = extracted_memory + input_embeddings

        # convert memory to log_probs
        out = self.fc1(extracted_memory)
        out = self.fc2(self.relu(out))
        return F.log_softmax(out, dim=-1)
    

    def fix_meta_embedding(self):
        # set the meta network to not require gradients
        for name, param in self.named_parameters():
            if name.startswith("embeddings"):
                param.requires_grad = False

    def unfix_meta_embedding(self):
        # set the meta network to not require gradients
        for name, param in self.named_parameters():
            if name.startswith("embeddings"):
                param.requires_grad = True

class MetaAttentionNetworkDirect(MetaAttentionNetwork):
    # query: batch_size * embed_dim
    def forward(self, query):
        if query.shape[-1] == self.n_classes:
            query = self.embeddings_query(query)
            input_embedings = torch.zeros(query.shape[0], self.memory_dim, device=query.device)
        else:
            query, input_embedings = torch.split(query, [self.memory_dim, query.shape[-1]-self.memory_dim], dim=-1)

        # memory
        keys = self.embeddings.T[:, :self.memory_dim] + self.embeddings_bias[:self.memory_dim]
        values = self.embeddings.T[:, self.memory_dim:] + self.embeddings_bias[self.memory_dim:]

        extracted_memory = attention(query, keys, values) + input_embedings

        # convert memory to log_probs
        out = self.fc1(extracted_memory)
        out = self.fc2(self.relu(out))
        return F.log_softmax(out, dim=-1)
    
    def remove_embeddings_query(self):
        self.embeddings_query.requires_grad_(False)
        self.n_margin = self.memory_dim - self.n_classes

class MetaClassNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, n_classes):
        super(MetaClassNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.embeddings = nn.Linear(n_classes, embed_dim)
        self.n_classes = n_classes

    def one_hot_class_embedding(self, one_hot):
        return self.embeddings(one_hot)
    
    def forward(self, x):

        if x.shape[-1] > self.n_classes:
            class_allocation, input_embeddings = torch.split(x, [self.n_classes, x.shape[-1]-self.n_classes], dim=-1)
            # embeddings = self.one_hot_class_embedding(torch.softmax(class_allocation, dim=-1))
            embeddings = self.one_hot_class_embedding(class_allocation)

            # class_allocation = class_allocation.reshape(original_shape[0], original_shape[1], -1)
            embeddings = input_embeddings + embeddings
            # embeddings = input_embeddings
        else:
            embeddings = self.one_hot_class_embedding(x)

        out = self.fc1(embeddings)
        out = self.fc2(self.relu(out))
        return F.log_softmax(out, dim=-1)
    
    def fix_meta_embedding(self):
        # set the meta network to not require gradients
        for name, param in self.named_parameters():
            if name.startswith("embeddings"):
                param.requires_grad = False
    

def attention(query, keys, values):
    # query: batch_size * embed_dim
    # keys: n_memories * embed_dim
    # values: n_memories * embed_dim
    scores = query.matmul(keys.transpose(0, 1))
    scores /= math.sqrt(query.shape[-1])
    scores = F.softmax(scores, dim=-1)
    output = scores.matmul(values)
    return output