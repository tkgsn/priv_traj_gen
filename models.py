import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import torch
import numpy as np
from opacus.layers.dp_rnn import DPGRUCell
from dataset import TrajectoryDataset
from my_utils import construct_default_quadtree


    
class BaseReferenceGenerator(nn.Module):
    """
    reference includes trajectory type and start_location
    this generates a trajectory that starts with start location and follows the trajectory type
    e.g., x, 1, 0 -> x, ?, x
    """

    def post_process(self, output, sampled, reference):
        """
        reference is a tuple of ints
        the length of the reference is the length of the trajectory
        the first element is the start location
        the rest of the elements are references to a first appearance of the location in the trajectory
        i.e., (x, 1, 0, 1) induces (x, y, x, y) where x and y are locations
        """
        # if the output is a list, this is the distributions for all layers
        output = output[-1] if type(output) == list else output
        assert len(reference) == output.shape[0], "the length of the reference should be {}, but it is {}".format(output.shape[0], len(reference))
        log_location_probs = output

        pointer = len(sampled)
        if pointer == 1:
            if reference[0][0] != -1:
                locations = torch.tensor([v[0] for v in reference]).view(-1, 1).to(output.device)
            elif torch.allclose(reference[:, 0], -1*torch.ones_like(reference[:, 0])):
                location_probs = torch.exp(log_location_probs).view(log_location_probs.shape[0], -1)
                locations = location_probs.multinomial(1)
            else:
                raise NotImplementedError("the first element of the reference should be -1, but it is {}".format(reference[0][0]))
        else:
            passed_locations = torch.concatenate([v for v in sampled[1:]], dim=1).view(-1, len(sampled)-1)
            log_location_probs = self.remove_location(log_location_probs, passed_locations)
            location_probs = torch.exp(log_location_probs).view(log_location_probs.shape[0], -1)
            # if all locations are removed, then we just sample from the first location
            indice = location_probs.sum(dim=-1) == 0
            location_probs[indice,0] = 1
            # sample
            locations = location_probs.multinomial(1)
            # replace by the reference
            locations = torch.concat([passed_locations, locations], dim=-1)[range(len(reference)), reference[:, pointer-1]].view(-1, 1)

        return locations
    
    def step(self, input):
        return self(input)

    def make_initial_input(self, reference):
        batch_size = len(reference)
        seq_len = max([len(ref) for ref in reference])
        outputs = torch.tensor([self.start_idx for _ in range(batch_size)]).view(batch_size, 1)
        return outputs, seq_len

    def concat(self, sampled, samples, reference):
        if samples == None:
            samples = []
        sampled = torch.concat(sampled, dim=1)
        samples.extend(sampled.cpu().detach().numpy().tolist())
        return samples

    def make_sample(self, references, batch_size):
        '''
        make n_samples of trajectories usign minibatch of batch_size
        kwargs are used as the auxiliary information to post-process output
        '''
        n_samples = len(references)

        samples = None
        for i in range(int(n_samples / batch_size)):
            reference = references[i*batch_size:(i+1)*batch_size]
            outputs, seq_len = self.make_initial_input(reference)
            reference_input = torch.tensor([list(ref) + [i for i in range(len(ref), seq_len)] for ref in reference])
            sampled = [outputs]
            for j in range(seq_len):
                outputs = self.step(outputs)
                # the main function of post_process is sample a location from the corresponding distribution
                outputs = self.post_process(outputs, sampled, reference_input)
                sampled.append(outputs)
            sampled = sampled[1:]

            samples = self.concat(sampled, samples, reference)
        
        return samples

    def remove_location(self, location, remove_locationss, log_prob=True):
        # remove the locations that are in remove_locations
        if remove_locationss is not None:
            assert location.shape[0] == len(remove_locationss)
            for i, remove_locations in enumerate(remove_locationss):
                if log_prob:
                    location[i, ..., remove_locations] = -float("inf")
                else:
                    location[i, ..., remove_locations] = 0
        if not log_prob:
            location = F.normalize(location, p=1, dim=-1)
        return location
    

class Markov1Generator(BaseReferenceGenerator):
    def __init__(self, transition_matrix, state_to_class):
        """
        transition_matrix: class * n_locations
        """
        n_classes = transition_matrix.shape[0]
        n_locations = len(state_to_class)
        assert transition_matrix.shape[1] == n_locations, "the number of locations should be {}, but it is {}".format(n_locations, transition_matrix.shape[1])
        start_prob = torch.zeros_like(transition_matrix[1]).view(1, -1)
        start_prob[0][0] = 1
        self.transition_matrix = torch.concat([transition_matrix, start_prob], dim=0)
        self.state_to_class = state_to_class
        self.start_idx = n_locations
        self.state_to_class[self.start_idx] = n_classes

    def __call__(self, input):
        """
        input is supposed to be batched location: batch_size * 1
        start_idx is going to be 0
        """
        classes = torch.tensor([self.state_to_class[state.item()] for state in input])
        output = self.transition_matrix[classes]
        return torch.log(output)
    
    def train(self):
        pass

    def eval(self):
        pass
    

class BaseTimeReferenceGenerator(BaseReferenceGenerator):
    """
    reference includes trajectory type and start_location
    this generates a trajectory that starts with start location and follows the trajectory type
    e.g., x, 1, 0 -> x, ?, x
    """

    def post_process(self, output, sampled, reference):
        """
        output is a tuple of (location, time)
        location is post processed by super().post_process
        time is sampled from the time distribution
        time starts with 1 and a next time should be larger than or equal to the previous time
        """
        
        log_location_probs, log_time_probs = output
        locations = super().post_process(log_location_probs, [v[0] for v in sampled], reference)
        if len(sampled) == 1:
            times = torch.ones_like(locations)
        else:
            # choose a time that is larger than the previous time
            previous_times = sampled[-1][1]
            passed_times = [list(range(time-1)) for time in previous_times]
            log_time_probs = self.remove_location(log_time_probs, passed_times)
            times = torch.exp(log_time_probs).multinomial(1)+1

        return [locations, times]
    
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


class GRUNet(BaseTimeReferenceGenerator):

    def __init__(self, n_locations, location_embedding_dim, time_dim, traj_type_dim, hidden_dim, reference_to_label):
        super(GRUNet, self).__init__()
        self.traj_type_embedding = nn.Embedding(traj_type_dim, hidden_dim)
        self.reference_to_label = reference_to_label
        self.n_locations = n_locations
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.gru = DPGRUCell(location_embedding_dim+time_dim, hidden_dim, True)
        self.fc = nn.Linear(hidden_dim, n_locations+time_dim-1)
        self.location_embedding = nn.Embedding(TrajectoryDataset.vocab_size(n_locations), location_embedding_dim)
        self.start_idx = TrajectoryDataset.start_idx(n_locations)
        self.relu = nn.ReLU()

    def make_initial_input(self, reference):
        batch_size = len(reference)
        seq_len = max([len(ref) for ref in reference])
        locations = torch.tensor([self.start_idx for _ in range(batch_size)]).to(next(self.parameters()).device).view(batch_size, 1)
        times = torch.tensor([0 for _ in range(batch_size)]).to(next(self.parameters()).device).view(batch_size, 1)
        states = self.init_hidden(reference)
        return [locations, times, states], seq_len

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
    
    # from the input, predict the next location probs and next time probs
    # input sequence fo (embed location, embed time): batch_size * seq_len * (embed_size+time_dim)
    # output location, time: seq_len * batch_size * (location_dim, time_dim)
    def forward(self, x, references):
        # embedding prefix
        out = self.input_gru(x, references)

        # decoding to probability of next location and next time
        location, time = self.decode(out)
        return location, time
    
    def init_hidden(self, references):
        # labels = torch.tensor([self.reference_to_label(reference) for reference in references]).to(next(self.parameters()).device)
        # print(labels)
        labels = torch.zeros(len(references), dtype=int).to(next(self.parameters()).device)
        hidden = self.traj_type_embedding(labels).view(-1, self.hidden_dim)
        return hidden
    
    def post_process(self, output, sampled, reference):

        log_location_probs, log_time_probs, states = output
        locations, times = super().post_process([log_location_probs, log_time_probs], [[v[0], v[1]] for v in sampled], reference)

        return [locations, times, states]
    
    def step(self, input):
        # input: [locations, times, states]
        next_state = self.encode(*input)
        location_probs, time_probs = self.decode(next_state)
        if type(location_probs) == list:
            location_probs = location_probs[-1]
        
        return [location_probs, time_probs, next_state]
    
    def encode(self, location, time, state):
        x = self.location_embedding(location).view(location.shape[0], -1)
        # one hot encode time
        times = F.one_hot(time, num_classes=self.time_dim).long().view(time.shape[0], -1)
        # concat time information
        x = torch.cat([x, times], dim=-1)

        state = self.gru(x, state)
        return state
    
    def decode(self, encoded):
        out = self.fc(self.relu(encoded))
        # split the last dimension into location and time
        log_location_probs, log_time_probs = torch.split(out, [self.n_locations, self.time_dim-1], dim=-1)
        log_location_probs = F.log_softmax(log_location_probs, dim=-1)
        log_time_probs = F.log_softmax(log_time_probs, dim=-1)

        return log_location_probs, log_time_probs



class MetaGRUNet(GRUNet):
    def __init__(self, meta_network, n_locations, location_embedding_dim, time_dim, traj_type_dim, hidden_dim, reference_to_label):
        super(MetaGRUNet, self).__init__(n_locations, location_embedding_dim, time_dim, traj_type_dim, hidden_dim, reference_to_label)
        
        self.meta_net = meta_network
        del self.location_embedding
        if hasattr(self.meta_net, "location_embedding"):
            # assert location_embedding_dim == self.meta_net.memory_dim, "location_embedding_dim should be {}, but it is {}".format(self.meta_net.memory_dim, location_embedding_dim)
            self.location_embedding = self.meta_net.location_embedding
        else:
            self.location_embedding = nn.Embedding(TrajectoryDataset.vocab_size(n_locations), location_embedding_dim)

        self.fc = None
        self.fc_location = nn.Linear(hidden_dim, self.meta_net.input_dim)
        self.fc_time = nn.Linear(hidden_dim, time_dim-1)
    
    def decode(self, embedding):
        out = self.fc_location(embedding)
        location = self.meta_net(out)
        # when meta net returns a list, it is multi-resolution task learning
        time = F.log_softmax(self.fc_time(embedding), dim=-1)
        return location, time

def compute_loss_meta_gru_net(target_locations, target_times, output_locations, output_times, coef_location, coef_time):
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

def compute_loss_gru_meta_gru_net(target_paths, target_times, output_locations, output_times, coef_location, coef_time):
    output_locations = output_locations.view(-1, 4)
    target_paths = target_paths.view(-1)
    loss_location = torch.nn.functional.nll_loss(output_locations, target_paths, ignore_index=4, reduction='mean') * coef_location
    loss_time = torch.nn.functional.nll_loss(output_times.view(-1, output_times.shape[-1]), target_times.view(-1)) * coef_time
    return loss_location, loss_time

    

class MetaNetwork(nn.Module):
    '''
    this class is a meta network, does not have any memory funtion
    that is, the hidden state is directory converted to log distribution on locations by MLP
    '''

    def __init__(self, memory_hidden_dim, hidden_dim, output_dim, n_classes, activate):
        '''
        output_dim := location_dim
        '''
        super(MetaNetwork, self).__init__()
        if activate == "relu":
            self.activate = nn.ReLU()
        elif activate == "leaky_relu":
            self.activate = nn.LeakyReLU()
        elif activate == "sigmoid":
            self.activate = nn.Sigmoid()
        else:
            raise NotImplementedError("activate should be relu or sigmoid, but it is {}".format(activate))

        self.input_dim = memory_hidden_dim
        self.query_to_scores = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), self.activate, nn.Linear(hidden_dim, output_dim))
        self.class_to_query = nn.Sequential(nn.Linear(n_classes, hidden_dim), self.activate, nn.Linear(hidden_dim, hidden_dim))
        self.hidden_to_query_ = nn.Sequential(nn.Linear(memory_hidden_dim, hidden_dim), self.activate, nn.Linear(hidden_dim, hidden_dim))
        self.n_classes = n_classes

    def forward(self, hidden):
        # hidden: batch_size * (seq_len) * n_classes
        hidden = hidden.view(hidden.shape[0], -1, hidden.shape[-1])
        
        query = self.hidden_to_query(hidden)
        # print(hidden.shape, query.shape)
        scores = self.compute_scores(query)
        return F.log_softmax(scores, dim=-1)
    
    def hidden_to_query(self, hidden):
       # for pre_training
        # if hidden.shape[-1] == self.n_classes:
        if hasattr(self, "class_to_query"):
            # hidden: batch_size * seq_len * n_classes
            query = self.class_to_query(hidden)
        else:
            query = self.hidden_to_query_(hidden)
        # output hidden: batch_size * seq_len * memory_dim
        return query

    def compute_scores(self, query):
        scores = self.query_to_scores(query)
        return scores
    
    def remove_class_to_query(self):
        del self.class_to_query
        # self.class_to_query.requires_grad_(False)


class BaseQuadTreeNetwork(nn.Module):
    def __init__(self, n_locations, memory_dim, hidden_dim, n_classes, activate, is_consistent=True):
        super(BaseQuadTreeNetwork, self).__init__()

        if activate == "relu":
            self.activate = nn.ReLU()
        elif activate == "leaky_relu":
            self.activate = nn.LeakyReLU()
        elif activate == "sigmoid":
            self.activate = nn.Sigmoid()
        else:
            raise NotImplementedError("activate should be relu or sigmoid, but it is {}".format(activate))

        self.tree = construct_default_quadtree(int(np.sqrt(n_locations))-2)
        self.tree.make_self_complete()
        self.memory_dim = memory_dim
        self.n_locations = len(self.tree.get_leafs())
        self.n_classes = n_classes
        self.class_to_query = nn.Linear(n_classes, self.memory_dim)
        # self.hidden_to_query_ = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), self.activate, nn.Linear(hidden_dim, self.memory_dim))
        self.hidden_to_query_ = nn.Linear(hidden_dim, self.memory_dim)
        self.is_consistent = is_consistent
        self.pre_training = True

    def forward(self, hidden):

        if len(hidden.shape) == 2:
            hidden = hidden.view(hidden.shape[0], 1, hidden.shape[1])
        
        query = self.hidden_to_query(hidden)
        # print(hidden.shape, query.shape)
        scores = self.compute_scores(query)
        distributions_for_all_depths = self.to_location_distribution(scores, target_depth=0)
        return distributions_for_all_depths
    
    def compute_scores(self, query):
        '''
        scores: batch_size * (seq_len) * (4^depth+4^depth-1+...+4)
        '''
        original_shape = query.shape
        keys = self.make_keys(query.shape)

        # matmal key and field
        query = query.view(original_shape[0] * original_shape[1], -1, self.memory_dim)
        keys = keys.view(original_shape[0] * original_shape[1], -1, self.memory_dim)
        scores = torch.bmm(keys, query.transpose(-2,-1)).view(*original_shape[:-1], -1, 4)
        return scores

    def hidden_to_query(self, hidden):
       # for pre_training
        # if hidden.shape[-1] == self.n_classes:
        if hasattr(self, "class_to_query"):
            query = self.class_to_query(hidden)
        else:
            query = self.hidden_to_query_(hidden)
        return query
    
    def make_states(self, key):
        '''
        convert key to keys of the nodes in the tree
        if target is given, keys are the nodes in the path that is determined by the target
        for each key, target can specify a path (list of inputs with length depth)
        therefore, the shape of the target should be the same as the shape of the input except the last dimension
        that is, when target = None -> output: batch_size * (seq_len) * (4^depth+4^depth-1+...+4) * memory_dim
        when target != None -> output: batch_size * (seq_len) * depth * memory_dim
        '''
        pass

    def make_keys(self, shape):
        '''
        convert state to key by self.state_to_key (MLP)
        shaep should be [batch_size, seq_len, memory_dim]
        output: batch_size * seq_len * (4^depth+4^depth-1+...+4) * memory_dim
        the key is aligned with the order of the node_id in the tree
        '''
        states = self.make_states(shape)

        cursor = 0
        keys = []
        for i in range(1,1+self.tree.max_depth):
            keys_i = self.state_to_key[i-1](states[...,cursor:cursor+4**i,:])
            keys.append(keys_i)
            cursor += 4**i

        keys = torch.cat(keys, dim=-2)
        return keys
    
    def to_location_distribution(self, scores, target_depth=-1):
        '''
        scores: batch_size * seq_len * n_nodes / 4 * 4
        scores is sorted according to node_id
        convert the output of forward without target to the log distribution for the depth
        if the depth is -1, the depth is the maximum depth of the tree
        if the depth is 0, return distributions of all layers (i.e., list of batch_size * seq_len * num_nodes at the depth)
        '''
        assert scores.shape[-2] == len(self.tree.get_all_nodes()) - len(self.tree.get_leafs()), "the range of the input distribution should be should be {}, but it is {}".format(len(self.tree.get_all_nodes()) - len(self.tree.get_leafs()), scores.shape[-2])
        if not self.training and self.pre_training:
            print("WARNING: PRE-TRAINING_MODE")
        # scores is sorted according to node.id
        def get_log_distribution_at_depth(scores, depth):

            if self.is_consistent:
                scores = F.log_softmax(scores, dim=-1)
                # when evaluation_mode, conducting consistent sampling for all locations; it generates probability distribution in all layers
                # that is, this generates Pr(location|depth)
                if (not self.training) or self.pre_training:
                # if True:
                    distribution = torch.zeros(*scores.shape[:-2], 4**depth).to(scores.device)
                    for i in range(depth):
                        ids = list(range(sum([4**depth_ for depth_ in range(0,i)]), sum([4**depth_ for depth_ in range(0,i+1)])))
                        score_at_depth_i = scores[...,ids,:].view(*scores.shape[:-2],-1).repeat_interleave(4**(depth-i-1), dim=-1)
                        distribution += score_at_depth_i
                    hidden_ids = list(range(sum([4**depth_ for depth_ in range(0,i+1)])-1, sum([4**depth_ for depth_ in range(0,i+2)])-1))
                    node_ids = [self.tree.hidden_id_to_node_id[id] for id in hidden_ids]
                    distribution = distribution[...,[id-node_ids[0] for id in node_ids]]
                    return distribution
                
            hidden_ids = list(range(scores.shape[-1] * scores.shape[-2]))
            node_ids = [self.tree.hidden_id_to_node_id[id] for id in hidden_ids]
            scores = scores.view(*scores.shape[:-2], -1)[..., [id-node_ids[0] for id in node_ids]]
            ids = list(range(sum([4**depth_ for depth_ in range(1,depth)]), sum([4**depth_ for depth_ in range(1,depth+1)])))
            if not self.is_consistent:
                # in the case of not consistent, this generates probability distribution on the depth layer
                distribution = F.log_softmax(scores[..., ids], dim=-1)
            else:
                # when training_mode, we consider that each node generates probability distribution on the 4 children nodes, i.e., P(child|parent)
                distribution = scores[..., ids]

            return distribution
        
        if target_depth == 0:
            return [get_log_distribution_at_depth(scores, depth) for depth in range(1, self.tree.max_depth+1)]
        elif target_depth == -1:
            target_depth = self.tree.max_depth
        
        return get_log_distribution_at_depth(scores, target_depth)
    
    def remove_class_to_query(self):
        # self.class_to_query.requires_grad_(False)
        self.pre_training = False
        del self.class_to_query



class LinearQuadTreeNetwork(BaseQuadTreeNetwork):
    def __init__(self, n_locations, memory_dim, hidden_dim, n_classes, activate, multilayer=False, is_consistent=False):
        super().__init__(n_locations, memory_dim, hidden_dim, n_classes, activate, is_consistent)
        if multilayer:
            self.linears = nn.ModuleList([nn.Sequential(nn.Linear(self.memory_dim, self.memory_dim), self.activate, nn.Linear(self.memory_dim, 4*self.memory_dim)) for _ in range(self.tree.max_depth)])
        else:
            self.linears = nn.ModuleList([nn.Linear(self.memory_dim, 4*self.memory_dim) for _ in range(self.tree.max_depth)])
        self.input_dim = hidden_dim
        # state_to_key is the standard MLP
        # self.state_to_key = nn.Sequential(nn.Linear(self.memory_dim, self.memory_dim), self.activate, nn.Linear(self.memory_dim, self.memory_dim))
        # self.state_to_key = nn.Linear(self.memory_dim, self.memory_dim)
        self.state_to_key = nn.ModuleList([nn.Sequential(nn.Linear(self.memory_dim, self.memory_dim), self.activate, nn.Linear(self.memory_dim, self.memory_dim)) for _ in range(self.tree.max_depth)])
        self.root_value = nn.Embedding(1, self.memory_dim)

    def make_states(self, shape):
        '''
        root_state: batch_size * seq_len * memory_dim
        output: batch_size * seq_len * (4^depth+4^depth-1+...+4) * memory_dim
        From the root_state, recursively apply the linear operation to the state until it reaches the maximum depth
        because the linear layer is (memory_dim, 4*memory_dim), each layer has 4 times parameters (nodes) than the previous layer
        '''
        # print(shape)
        device = self.root_value.weight.device
        states = []
        ith_state = self.root_value(torch.zeros(*shape[:-1], device=device).long())
        for linear in self.linears:
            ith_state = linear(ith_state).view(ith_state.shape[0], ith_state.shape[1], -1, self.memory_dim)
            states.append(ith_state)
        states = torch.concat(states, dim=-2)

        return states

# in this class, location embedding comes from the tconvs with input of the self.root_value(1)
# this class requires privtree
class FullLinearQuadTreeNetwork(LinearQuadTreeNetwork):
    def __init__(self, n_locations, memory_dim, hidden_dim, location_embedding_dim, privtree, activate, multilayer, is_consistent):
        # n_classes = len(privtree.get_leafs())
        self.location_embedding_dim = location_embedding_dim
        n_classes = len(privtree.merged_leafs)
        self.n_locations = n_locations
        super().__init__(n_locations, memory_dim, hidden_dim, n_classes, activate, multilayer, is_consistent)
        self.privtree = privtree
        self.root_value = nn.Embedding(3, self.memory_dim)
        # 0 -> states, 1 -> start value, 2 -> ignore value
        self.leaf_ids = self.privtree.get_leaf_ids_in_tree(self.tree)
        # self.hidden_ids = torch.tensor([self.tree.node_id_to_hidden_id[node_id] for node_id in self.leaf_ids])
        self.node_ids = torch.tensor(self.privtree.get_leaf_ids_in_tree(self.tree))
        # self.class_to_query = nn.Linear(location_embedding_dim, self.memory_dim)
        self.class_to_query = nn.ModuleList([nn.Linear(location_embedding_dim, self.memory_dim) for _ in range(self.tree.max_depth)])
        # self.class_to_query = nn.Sequential(nn.Linear(location_embedding_dim, self.memory_dim), self.activate, nn.Linear(self.memory_dim, self.memory_dim))
        self.state_to_location_embedding = nn.Linear(self.memory_dim, location_embedding_dim)

        # self.hidden_to_query_ = nn.ModuleList([nn.Linear(hidden_dim, self.memory_dim) for _ in range(self.tree.max_depth)])
        self.hidden_to_query_ = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, self.memory_dim), self.activate, nn.Linear(self.memory_dim, self.memory_dim)) for _ in range(self.tree.max_depth)])

    def compute_scores(self, querys):
        '''
        scores: batch_size * (seq_len) * (4^depth+4^depth-1+...+4)
        '''
        keys = self.make_keys(querys[0].shape)

        scores = []

        for key, query in zip(keys, querys):
            original_shape = query.shape
            # matmal key and field
            query = query.view(original_shape[0] * original_shape[1], -1, self.memory_dim)
            key = key.view(original_shape[0] * original_shape[1], -1, self.memory_dim)
            scores.append(torch.bmm(key, query.transpose(-2,-1)).view(*original_shape[:-1], -1, 4))
        
        scores = torch.cat(scores, dim=-2)
        return scores

    def hidden_to_query(self, hidden):
       # for pre_training
        if hasattr(self, "class_to_query"):
            # hidden: batch_size * seq_len * n_classes
            node_embeddings = self.location_embedding(self.node_ids.to(hidden.device), True) # n_classes * memory_dim_
            node_embeddings_ = torch.zeros(self.n_classes, self.location_embedding_dim, device=hidden.device)
            for i, leafs in enumerate(self.privtree.merged_leafs):
                for leaf in leafs:
                    index = self.leaf_ids.index(leaf.id)
                    node_embeddings_[i] += node_embeddings[index]

            location_embeddings = hidden.matmul(node_embeddings_) # batch_size * memory_dim

            querys = []
            for linear in self.class_to_query:
                querys.append(linear(location_embeddings))
        else:
            querys = []
            for linear in self.hidden_to_query_:
                querys.append(linear(hidden))
        # output hidden: batch_size * seq_len * memory_dim
        return querys
    
    def location_embedding(self, location, is_node_id=False):
        # virtual input for grad_sample
        batch_size = location.shape[0]
        start_value = self.root_value(torch.ones(batch_size, device=location.device).long()).view(batch_size, -1, 1, self.memory_dim)
        ignore_value = self.root_value(torch.ones(batch_size, device=location.device).long()*2).view(batch_size, -1, 1, self.memory_dim)
        states = self.make_states([batch_size, 1, 1])
        # add the start value to states
        states = torch.cat([states, start_value], dim=-2)
        # add the ignore value to states
        states = torch.cat([states, ignore_value], dim=-2)

        if not is_node_id:
            # input is a state, which is in the order of from upper left to lower right
            # but we need to access to location embedding which is sorted according to node.id
            location_ = []
            for loc in location.view(-1).tolist():
                if loc < self.n_locations:
                    location_.append(self.tree.state_to_node_id_path(loc)[-1]-1)
                else:
                    location_.append(loc)
            location = location_
        else:
            location = location -1
            # because the first node is the root node and make_states does not include the root node
            location = location.view(-1).tolist()
        states = states[list(range(batch_size)), ..., location, :]
        location_embeddings = self.state_to_location_embedding(states).view(batch_size, -1)
        return location_embeddings
    
    def make_keys(self, shape):
        '''
        convert state to key by self.state_to_key (MLP)
        shaep should be [batch_size, seq_len, memory_dim]
        output: batch_size * seq_len * (4^depth+4^depth-1+...+4) * memory_dim
        the key is aligned with the order of the node_id in the tree
        '''
        states = self.make_states(shape)

        cursor = 0
        keys = []
        for i in range(1,1+self.tree.max_depth):
            keys_i = self.state_to_key[i-1](states[...,cursor:cursor+4**i,:])
            keys.append(keys_i)
            cursor += 4**i

        return keys
    


def guide_to_model(type):
    if type == "fulllinear_quadtree":
        return FullLinearQuadTreeNetwork, MetaGRUNet
    elif type == "meta_network":
        return MetaNetwork, MetaGRUNet