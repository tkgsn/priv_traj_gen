import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import torch
import numpy as np
from opacus.layers.dp_rnn import DPGRUCell
from dataset import TrajectoryDataset
from my_utils import construct_default_quadtree, depth_clustering
from abc import ABCMeta, abstractmethod

class Generator(nn.Module):
    def __init__(self, location_encoding_component, time_encoding_component, prefix_encoding_component, scoring_component):
        super(Generator, self).__init__()
        self.location_encoding_component = location_encoding_component
        self.time_encoding_component = time_encoding_component
        self.prefix_encoding_component = prefix_encoding_component
        self.scoring_component = scoring_component

    def forward(self, x, states=None):
        locations = x[0]
        times = x[1]
        # encoding of each point (location, time)
        location_embedding = self.location_encoding_component(locations)
        time_embedding = self.time_encoding_component(times)
        embedding_sequence = torch.cat([location_embedding, time_embedding], dim=-1)

        # encoding of prefix (embedding_of_point, embedding_of_point, ...)
        hiddens, prefix_embedding = self.prefix_encoding_component(embedding_sequence, states)

        # decoding to scores as probability of next location and next time
        location, time = self.scoring_component(hiddens)
        return [location, time], hiddens

    # for pre-training
    def transition(self, class_id, class_encoder, temp_prefix_encoding_component):
        class_embedding = class_encoder(class_id)
        prefix_embedding = temp_prefix_encoding_component(class_embedding)
        location, _ = self.scoring_component(prefix_embedding)
        return location

    def make_sample(self, references, time_references, batch_size):
        '''
        make n_samples of trajectories usign minibatch of batch_size
        '''
        n_samples = len(references)

        samples = None
        for i in range(int(n_samples / batch_size)):
            reference = references[i*batch_size:(i+1)*batch_size]
            time_reference = time_references[i*batch_size:(i+1)*batch_size]

            batch_size = len(reference)
            seq_len = max([len(ref) for ref in reference])
            
            # make initial input
            locations = torch.tensor([[self.location_encoding_component.start_idx(), reference[i][0]] for i in range(batch_size)]).to(next(self.parameters()).device).view(batch_size, 2)
            times = torch.tensor([[0,time_reference[i]] for _ in range(batch_size)]).to(next(self.parameters()).device).view(batch_size, 2)
            prefix_embedding = None

            # recurrently sample next location and time
            sampled = [(locations[:,1].view(-1, 1), times[:,1].view(-1, 1))]
            for _ in range(seq_len):
                (locations, times), hiddens = self([locations, times], prefix_embedding)
                # this is post processing for consistent generation
                locations = self.scoring_component.to_location_distribution(locations)

                # sampling from the mutinomial distribution
                locations = torch.exp(locations).view(locations.shape[0], -1).multinomial(1)
                times = torch.exp(times[:,-1,:]).multinomial(1)

                sampled.append((locations, times))
                prefix_embedding = hiddens[:, -1, :]

            samples = self.concat(sampled, samples, reference)
        return samples

    def concat(self, sampled, samples, reference):
        if samples == None:
            samples = [[], []]
        locations = torch.concat([v[0] for v in sampled], dim=1).cpu().detach().numpy().tolist()
        times = torch.concat([v[1] for v in sampled], dim=1).cpu().detach().numpy().tolist()

        # remove the outside of the format
        for i in range(len(locations)):
            samples[0].append(locations[i][:len(reference[i])])
            samples[1].append(times[i][:len(reference[i])])
        
        return samples


# 
class LocationEncodingComponent(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def make_class_encoder(self):
        pass

    def start_idx(self):
        return TrajectoryDataset.start_idx(self.n_locations)

class MatrixLocationEncodingComponent(LocationEncodingComponent):
    def __init__(self, n_locations, dim):
        super(MatrixLocationEncodingComponent, self).__init__()
        self.embedding_matrix = nn.Embedding(TrajectoryDataset.vocab_size(n_locations), dim)
        self.dim = dim
        self.n_locations = n_locations


    def forward(self, location):
        return self.embedding_matrix(location)

    # making a compatible temporary component which encodes class for pre-training
    # warning: this is not currently trainable
    def make_class_encoder(self, privtree):
        class ClassEncoder(nn.Module):

            def __init__(self_):
                super(ClassEncoder, self_).__init__()
                self_.dim = len(privtree.get_leafs())

            def forward(self_, class_hidden_vector):
                return class_hidden_vector
            
        return ClassEncoder()
        

class LinearHierarchicalLocationEncodingComponent(LocationEncodingComponent):
    def __init__(self, n_locations, dim):
        super(LinearHierarchicalLocationEncodingComponent, self).__init__()

        self.tree = construct_default_quadtree(int(np.sqrt(n_locations))-2)
        self.tree.make_self_complete()
        # self.n_locations = len(self.tree.get_leafs())
        self.n_locations = n_locations

        self.special_vectors = nn.Embedding(1+TrajectoryDataset.n_specials(), dim)
        # 0: root, 1: start, 2: ignore, 3: end (refer to dataset)
        self.linears = nn.ModuleList([nn.Linear(dim, 4*dim) for _ in range(self.tree.max_depth)])
        self.dim = dim

    def location_to_index(self, location, depth):

        def location_to_index_with_special_ids(location):
            if location >= self.n_locations and location < self.n_locations+TrajectoryDataset.n_specials():
                special_vocab_id = location - self.n_locations + 1
                index = self.tree.state_to_node_id_path(self.n_locations-1)[depth] + special_vocab_id
            else:
                node_id = self.tree.state_to_node_id_path(location)[depth]
                index = self.tree.node_id_to_hidden_id[node_id]
            return index

        indices = location.cpu().detach().clone()
        indices.apply_(lambda x: location_to_index_with_special_ids(x))
        
        # # change the order according to the geographical order ()
        # hidden_ids = indices.apply_(lambda x: node_to_hidden(x))

        return indices
    
    def make_embedding_matrix(self, batch_size, device):
        # make root state
        temp_input = torch.tensor(list(range(self.special_vectors.num_embeddings))*batch_size).view(batch_size, -1).to(device)
        root_state = self.special_vectors(temp_input)
        ith_state = root_state[:,0,:].view(batch_size, -1, self.dim)

        # deconvolutional operation
        states = [ith_state]
        for linear in self.linears:
            ith_state = linear(ith_state).view(batch_size, -1, self.dim)
            states.append(ith_state)
        states = torch.concat(states, dim=-2)

        # add special vocabs in the last
        states = torch.concat([states, root_state[:,1:,:]], dim=-2)
        return states

    def forward(self, location, depth=-1):
        batch_size = location.shape[0]
        states = self.make_embedding_matrix(batch_size, location.device)

        # fetch the location embedding from the results
        indices = self.location_to_index(location, depth)
        location_embeddings = []
        for i in range(batch_size):
            location_embeddings.append(states[i][indices[i],:])
        location_embeddings = torch.stack(location_embeddings, dim=0)
        return location_embeddings

    # making a compatible temporary component which encodes class for pre-training
    def make_class_encoder(self, privtree):

        # fetch the node ids of the privtree leafs (=classes)
        node_ids = privtree.get_leaf_ids_in_tree(self.tree)

        class ClassEncoder(nn.Module):

            def __init__(self_):
                super(ClassEncoder, self_).__init__()
                self_.dim = self.dim

            def forward(self_, class_hidden_vector):

                # encode nodes
                embedding_matrix = self.make_embedding_matrix(1, class_hidden_vector.device)[0]
                node_embeddings = embedding_matrix[node_ids]

                # merge the node embeddings with
                node_embeddings_ = torch.zeros(len(node_ids), self.dim, device=class_hidden_vector.device)
                for i, leafs in enumerate(privtree.merged_leafs):
                    for leaf in leafs:
                        index = node_ids.index(leaf.id)
                        node_embeddings_[i] += node_embeddings[index]

                # merge the node embeddings with class_hidden_vector (=input ratio)
                class_embeddings = class_hidden_vector.matmul(node_embeddings_) # batch_size * memory_dim
                return class_embeddings
        
        return ClassEncoder()
        

class TimeEncodingComponent(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, time):
        pass

class MatrixTimeEncodingComponent(TimeEncodingComponent):
    def __init__(self, n_times, dim):
        super(MatrixTimeEncodingComponent, self).__init__()
        self.embedding_matrix = nn.Embedding(TrajectoryDataset.time_vocab_size(n_times), dim)

    def forward(self, time):
        return self.embedding_matrix(time)


class PrefixEncodingComponent(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x):
        pass

    # making a compatible temporary network which works as substitution of prefix_encoding_component for pre-training
    @abstractmethod
    def make_temp_network(self, n_classes):
        pass
    

class GRUPrefixEncodingComponent(PrefixEncodingComponent):
    def __init__(self, input_dim, hidden_dim, n_layers, bidirectional):
        super(GRUPrefixEncodingComponent, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = DPGRUCell(input_dim, hidden_dim, True)

    def forward(self, embedding_sequence, hidden_states=None):
        batch_size, seq_len, _ = embedding_sequence.shape

        # make initial hidden states
        if hidden_states is not None:
            hidden = hidden_states
        else:
            hidden = torch.zeros(batch_size, self.hidden_dim).to(embedding_sequence.device)

        # recurrently encode the prefix
        hiddens = []
        for i in range(seq_len):
            hidden = self.gru_cell(embedding_sequence[:,i,:], hidden)
            hiddens.append(hidden)
        hiddens = torch.stack(hiddens, dim=1)
        return hiddens, hidden

    # making a compatible temporary network which works as substitution of prefix_encoding_component for pre-training
    def make_temp_network(self, input_dim):
        return nn.Linear(input_dim, self.hidden_dim)
    

class ScoringComponent(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, prefix_embedding):
        pass

    # locations is expected to be the shape of batch_size * seq_len * n_locations and the output is the log distribution of the next location
    def to_location_distribution(self, locations):
        return locations[:,-1,:]

class LinearScoringComponent(ScoringComponent):
    def __init__(self, hidden_dim, n_locations, n_times):
        super(LinearScoringComponent, self).__init__()
        self.fc_location = nn.Linear(hidden_dim, n_locations)
        self.fc_time = nn.Linear(hidden_dim, n_times)
        self.multitask = False

    def forward(self, prefix_embedding):
        location = F.log_softmax(self.fc_location(prefix_embedding), dim=-1)
        time = F.log_softmax(self.fc_time(prefix_embedding), dim=-1)
        return location, time

class DotScoringComponent(ScoringComponent):
    def __init__(self, hidden_dim, n_locations, n_times, location_encoding_component, multitask, consistent):
        super(DotScoringComponent, self).__init__()
        self.n_locations = n_locations
        self.fc_time = nn.Linear(hidden_dim, n_times)
        self.location_encoding_component = location_encoding_component
        self.embedding_to_key = nn.Linear(location_encoding_component.dim, hidden_dim)
        self.multitask = multitask
        self.consistent = consistent

    def forward(self, prefix_embedding):
        batch_size = prefix_embedding.shape[0]
        keys = self.make_keys(batch_size, prefix_embedding.device)

        # compute score by dot product of prefix_embedding (=query) and keys
        scores = torch.bmm(prefix_embedding.view(batch_size, -1, prefix_embedding.shape[-1]), keys.transpose(-2,-1))
        
        # fetch the scores of the locations and log_softmax
        all_locations = torch.tensor(range(self.n_locations))

        # for multi-resolution task learning, compute the scores of the nodes at all depths
        # otherwise, compute the scores of the nodes at the deepest depth
        depths = range(1,self.location_encoding_component.tree.max_depth+1) if self.multitask else [-1]

        location_ = []
        for depth in depths:
            # convert location to the corresponding location_ids of the depth
            location_ids = self.location_encoding_component.location_to_index(all_locations, depth).tolist()
            # remove duplicate values and sort
            location_ids = list(set(location_ids))
            location_ids.sort()

            # fetch the scores of the nodes
            scores_ = []
            for i in range(batch_size):
                scores_.append(scores[i][..., location_ids])
            
            # convert to distribution
            location_.append(F.log_softmax(torch.stack(scores_, dim=0), dim=-1))

        location = location_ if self.multitask else location_[-1]
        time = F.log_softmax(self.fc_time(prefix_embedding), dim=-1)
        return location, time

    def make_keys(self, batch_size, device):
        embedding_matrix = self.location_encoding_component.make_embedding_matrix(batch_size, device)
        return self.embedding_to_key(embedding_matrix)

    # this converts the output of forward to the log distribution of the next location at the deepest depth
    def to_location_distribution(self, locations):
        if self.multitask:
            if self.consistent:
                assert type(locations) == list, "the output should be a list, but it is {}".format(type(locations))

                # fetch the next location distributions for all depths
                locations = [location[:,-1,:] for location in locations]
                depth = len(locations)

                # compute the distribution at the deepest depth reccurently from the above distributions
                distribution = torch.zeros_like(locations[-1]).to(locations[-1].device)
                for i in range(depth):
                    score_at_depth_i = locations[i].repeat_interleave(4**(depth-i-1), dim=-1)
                    distribution += score_at_depth_i
                return distribution
            else:
                return locations[-1][:,-1,:]
        else:
            return super().to_location_distribution(locations)


def compute_loss_generator(target_locations, target_times, output_locations, output_times, coef_location, coef_time):
    # list type means multi-resolution task learning
    if type(target_locations) != list:
        output_locations = [output_locations[-1]] if type(output_locations) == list else [output_locations]
        target_locations = [target_locations]

    n_locations = output_locations[-1].shape[-1]
    loss = []
    for i in range(len(target_locations)):
        location_dim = output_locations[i].shape[-1]
        output_locations[i] = output_locations[i].view(-1, location_dim)
        target_locations[i] = target_locations[i].view(-1)
        # coef = (i+1)/len(target_locations)
        coef = 1
        loss.append(coef*F.nll_loss(output_locations[i], target_locations[i], ignore_index=TrajectoryDataset.ignore_idx(n_locations)) * coef_location)
    loss.append(F.nll_loss(output_times.view(-1, output_times.shape[-1]), (target_times).view(-1)) * coef_time)
    return loss


def construct_generator(model_name, n_locations, n_times, location_embedding_dim, time_embedding_dim, memory_hidden_dim, multitask, consistent):

    time_encoding_component = MatrixTimeEncodingComponent(n_times-1, time_embedding_dim)
    input_dim = location_embedding_dim + time_embedding_dim
    prefix_encoding_component = GRUPrefixEncodingComponent(input_dim, memory_hidden_dim, 1, False)
    
    if model_name == "baseline":
        location_encoding_component = MatrixLocationEncodingComponent(n_locations, location_embedding_dim)
        scoring_component = LinearScoringComponent(memory_hidden_dim, n_locations, n_times)
    elif model_name == "hrnet":
        location_encoding_component = LinearHierarchicalLocationEncodingComponent(n_locations, location_embedding_dim)
        scoring_component = DotScoringComponent(memory_hidden_dim, n_locations, n_times, location_encoding_component, multitask, consistent)
    
    generator = Generator(location_encoding_component, time_encoding_component, prefix_encoding_component, scoring_component)
    return generator