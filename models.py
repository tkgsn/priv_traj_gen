import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import torch
import numpy as np
from opacus.layers.dp_rnn import DPGRU
from dataset import TrajectoryDataset

class MetaDiscreteTimeTrajTypeGRUNet(nn.Module):
    def __init__(self, input_dim, traj_type_dim, hidden_dim, output_dim, time_dim, n_layers, embed_size, drop_prob=0.1):
        super(MetaDiscreteTimeTrajTypeGRUNet, self).__init__()
        self.traj_type_embedding = nn.Embedding(traj_type_dim, hidden_dim*n_layers)
        # self.traj_type_embedding = nn.Embedding(traj_type_dim, 10)
        # self.fc_traj_type = nn.Linear(10, hidden_dim*n_layers)
        self.embeddings = nn.Embedding(input_dim, embed_size)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.time_dim = time_dim
        self.output_dim = output_dim
        
        self.gru = DPGRU(embed_size+time_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        # output_dim + 1 because we also want to predict the time
        self.fc = nn.Linear(hidden_dim, output_dim+time_dim)
        self.relu = nn.ReLU()


    def input_gru(self, x, labels):
        locations = x[0]
        times = x[1]

        h = self.init_hidden(labels)
        x = self.embeddings(locations)

        # one hot encode time
        times = F.one_hot(times, num_classes=self.time_dim).long().view(times.shape[0], times.shape[1], -1)
        
        # concat time information
        x = torch.cat([x, times], dim=-1)

        out, _ = self.gru(x, h)

        return out

    def forward(self, x, labels):

        out = self.input_gru(x, labels)

        out = self.fc(self.relu(out))
        location = out[:, :, :-self.time_dim]
        location = F.log_softmax(location, dim=-1)
        time = out[:, :, -self.time_dim:]
        time = F.log_softmax(time, dim=-1)
        return location, time
        

    def forward_log_prob(self, x):
        return self(x)

    # def init_hidden(self, batch_size):
    #     batch_size = len(batch_size)
    #     weight = next(self.parameters()).data
    #     hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(next(self.parameters()).device)
    #     return hidden

    def init_hidden(self, labels):
        hidden = self.traj_type_embedding(labels).view(self.n_layers, -1, self.hidden_dim)
        # hidden = self.fc_traj_type(hidden).view(self.n_layers, -1, self.hidden_dim)
        return hidden

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

class MetaClassNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, n_classes):
        super(MetaClassNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        # self.embeddings = nn.Embedding(n_classes, embed_dim)
        self.embeddings = nn.Linear(n_classes, embed_dim)
        # set the required_grads to false so that the embeddings are not updated
        # self.embeddings.weight.requires_grad = False
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
    

def attention(query, keys, values):
    # query: batch_size * embed_dim
    # keys: n_memories * embed_dim
    # values: n_memories * embed_dim
    scores = query.matmul(keys.transpose(0,1))
    scores /= math.sqrt(query.shape[-1])
    # batch_size * n_memories

    scores = F.softmax(scores, dim = -1)
    output = scores.matmul(values)
    # batch_size * embed_dim
    return output

class MetaAttentionNetwork(nn.Module):
    """
    This network is pre-trained using the probability distributions
    This intends to implicitly memory n_memories distributions by self.embeddings

    input: query := (batch_size * embed_dim)
    output: log_probs := (batch_size * n_locations)
    """
    
    def __init__(self, embed_dim, hidden_dim, n_locations, n_classes):
        super(MetaAttentionNetwork, self).__init__()
        # embedding with name "embed"
        # self.embeddings = nn.Embedding(n_classes, embed_dim*2)
        self.embeddings = nn.Linear(n_classes, embed_dim*2)
        self.n_classes = n_classes
        self.embed_dim = embed_dim

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_locations)
        self.relu = nn.ReLU()
        self.ignore_input_embeddings = False

    def forward(self, query):
        # memory
        # input_to_memory = torch.tensor([i for i in range(self.n_classes)]).to(query.device)
        input_to_memory = torch.eye(self.n_classes).to(query.device)
        kvs = self.embeddings(input_to_memory)
        keys = kvs[:, :self.embed_dim]
        values = kvs[:, self.embed_dim:]

        if query.shape[-1] > self.n_classes:
            query, input_embeddings = torch.split(query, [self.n_classes, query.shape[-1]-self.n_classes], dim=-1)
        else:
            input_embeddings = torch.zeros(query.shape[0], self.embed_dim, device=query.device)
        
        query = query.matmul(keys)
        extracted_memory = attention(query, keys, values)
        extracted_memory = extracted_memory + input_embeddings * (1-self.ignore_input_embeddings)

        # convert memory to log_probs
        out = self.fc1(extracted_memory)
        out = self.fc2(self.relu(out))
        return F.log_softmax(out, dim=-1)
        
    
class MetaGRUNet(nn.Module):
    def __init__(self, meta_network, input_dim, traj_type_dim, hidden_dim, output_dim, time_dim, n_layers, embed_size, fix_meta_network, fix_embedding, drop_prob=0.1):
        super(MetaGRUNet, self).__init__()
        # self.traj_type_embedding = nn.Embedding(traj_type_dim, 10)
        # self.fc_traj_type = nn.Linear(10, hidden_dim*n_layers)
        self.embeddings = nn.Embedding(input_dim, embed_size)
        # self.embeddings = nn.Linear(input_dim, embed_size)
        self.hidden_dim = hidden_dim+meta_network.n_classes
        self.traj_type_embedding = nn.Embedding(traj_type_dim, self.hidden_dim*n_layers)
        self.n_layers = n_layers
        self.time_dim = time_dim
        self.output_dim = output_dim
        
        self.gru = DPGRU(embed_size+time_dim, self.hidden_dim, n_layers, batch_first=True, dropout=drop_prob)

        # set the meta network to not require gradients
        for name, param in meta_network.named_parameters():
            param.requires_grad = not fix_meta_network
            if name in ["embeddings.weight", "embeddings.bias"] and fix_embedding:
                param.requires_grad = not fix_embedding
        # self.weight = meta_network.embeddings.weight
        self.meta_net = meta_network
        self.fc = nn.Linear(self.hidden_dim, time_dim)

        self.fc_meta1 = nn.Linear(self.hidden_dim, 100)
        self.fc_meta2 = nn.Linear(100, self.hidden_dim)

        self.relu = nn.ReLU()

    def set_requires_grad_of_meta_network(self, requires_grad):
        for param in self.meta_net.parameters():
            param.requires_grad = requires_grad
        self.meta_net.ignore_input_embeddings = not requires_grad

    def input_gru(self, x, labels):
        locations = x[0]
        times = x[1]

        h = self.init_hidden(labels)
        # one_hot encoding of locations
        # locations = F.one_hot(locations, num_classes=self.output_dim).long().view(locations.shape[0], locations.shape[1], -1)
        x = self.embeddings(locations)

        # one hot encode time
        times = F.one_hot(times, num_classes=self.time_dim).long().view(times.shape[0], times.shape[1], -1)
        
        # concat time information
        x = torch.cat([x, times], dim=-1)
        out, _ = self.gru(x, h)

        return out

    def embedding(self, x, labels):
        out = self.input_gru(x, labels)
        out = self.fc_meta1(out)
        out = self.fc_meta2(self.relu(out))
        return out

    def forward(self, x, labels):

        out = self.embedding(x, labels)
        location = self.meta_net(out)
        time = self.fc(out)
        time = F.log_softmax(time, dim=-1)
        return location, time
        

    def forward_log_prob(self, x):
        return self(x)

    def init_hidden(self, labels):
        hidden = self.traj_type_embedding(labels).view(self.n_layers, -1, self.hidden_dim)
        # hidden = self.fc_traj_type(hidden).view(self.n_layers, -1, self.hidden_dim)
        return hidden
    

def step(generator, i, output_dim, sample, sample_time, labels, start_index, end_index, ignore_index, without_time):
    input = sample.to(next(generator.parameters()).device, non_blocking=True)[:, :i+1].long()
    input_time = sample_time.to(next(generator.parameters()).device, non_blocking=True)[:, :i+1].long()
    labels = labels.to(next(generator.parameters()).device, non_blocking=True)
    input[input==end_index] = ignore_index
    pred_location, pred_time = generator([input, input_time], labels)
    probs = torch.exp(pred_location).detach().cpu().numpy()
    probs = probs[:,-1]
    probs = probs[:,:output_dim] / probs[:,:output_dim].sum(axis=1, keepdims=True)

    time_probs = torch.exp(pred_time).detach().cpu().numpy()
    time_probs = time_probs[:,-1]
    time_output_dim = time_probs.shape[1]

    for j, prob in enumerate(probs):
        if sample[j, i] == end_index:
            choiced = end_index
        else:
            for pre_location_index in range(1,i+1):
                pre_location = int(sample[j, pre_location_index].item())
                if pre_location < output_dim:
                    prob[pre_location] = 0
            prob = prob / prob.sum()
            choiced = np.random.choice(output_dim, p=prob)
        sample[j, i+1] = choiced

    for j, prob in enumerate(time_probs):
        pre_time = int(sample_time[j, i].item())
        if without_time:
            choiced = pre_time + 1
        else:
            # replace the prob of the previous time step with 0
            # it is not possible that the model outputs the 0th time step because the 0th time step represents time [-inf, 0]
            prob[0] = 0
            prob[:pre_time] = 0
            prob = prob / prob.sum()
            choiced = np.random.choice(time_output_dim, p=prob)
        sample_time[j, i+1] = choiced

    return sample, sample_time


def make_frame_data(n_sample, seq_len, start_index):
    frame_data = torch.zeros((n_sample, seq_len+1))
    frame_data.fill_(start_index)
    return frame_data


def recurrent_step(generator, seq_len, output_dim, start_time, data, time_data, labels, start_index, end_index, ignore_index, label_to_format, without_time):
    # convert label (format) to reference
    # reference refers to the index of the first appearance of a state
    # if the label stands for format "010202", the reference is "010303"
    def make_reference(label):
        format = label_to_format[label.item()]
        reference = {}
        for i in range(len(format)):
            if format[i] not in reference:
                reference[format[i]] = i
        return [reference[format[i]] for i in range(len(format))]
    references = [make_reference(label) for label in labels]

    for i in range(start_time, seq_len):
        data, time_data = step(generator, i, output_dim, data, time_data, labels, start_index, end_index, ignore_index, without_time)
        # apply fix_by_label to each record
        # print(references)
        for index, reference in enumerate(references):
            # if data[index][1].item() == 2978:
            #     print(reference, len(references))
            #     print(index, data[index], label_to_format[labels[index].item()])
            if len(reference) <= i:
                data[index][i+1] = end_index
                continue
            data[index][i+1] = data[index][reference[i]+1]
    # for label, record in zip(labels, data):
    #     if record[1].item() == 2978:
    #         print(label_to_format[label.item()], record)
    return data[:,1:], time_data


def make_sample(batch_size, generator, labels, label_to_format, n_locations, time_end_index, real_start=None, without_time=False):
    seq_len = max([len(v) for v in label_to_format.values()])
    start_idx = TrajectoryDataset.start_idx(n_locations)
    end_idx = TrajectoryDataset.end_idx(n_locations)
    ignore_idx = TrajectoryDataset.ignore_idx(n_locations)

    n_sample = len(labels)
    frame_data = make_frame_data(n_sample, seq_len, start_idx)
    frame_data_for_time = make_frame_data(n_sample, seq_len, 0)
    start_time = 0

    if real_start is not None:
        indice = np.random.choice(range(len(real_start[0])), n_sample, replace=False)
        # choose real data by indice from list
        real_traj = real_start[0]
        real_traj = [real_traj[i] for i in indice]
        real_time_traj = real_start[1]
        real_time_traj = [real_time_traj[i] for i in indice]
        frame_data[:,1] = torch.tensor([v[0] for v in real_traj])
        frame_data_for_time[:,1] = torch.tensor([v[1] for v in real_time_traj])
        start_time = 1
    
    samples = []
    time_samples = []
    for i in range(int(n_sample / batch_size)):
        batch_labels = labels[i*batch_size:(i+1)*batch_size]
        batch_input = frame_data[i*batch_size:(i+1)*batch_size]
        batch_input_for_time = frame_data_for_time[i*batch_size:(i+1)*batch_size]
        sample, time_sample = recurrent_step(generator, seq_len, n_locations, start_time, batch_input, batch_input_for_time, batch_labels, start_idx, end_idx, ignore_idx, label_to_format, without_time)
        
        sample = sample.cpu().detach().long().numpy()
        time_sample = time_sample.cpu().detach().numpy()

        samples.extend(sample)
        time_samples.extend(time_sample)

    # print(samples)
    generated_trajectories = []
    for sample in samples:
        generated_trajectories.append([v for v in sample if v != end_idx])

    generated_time_trajectories = []
    for sample in time_samples:
        generated_time_trajectories.append([v for v in sample if v != ignore_idx])

    for label, i in zip(labels,range(len(generated_time_trajectories))):
        format = label_to_format[label.item()]
        # print(i, format, generated_time_trajectories[i])
        generated_time_trajectories[i][len(format)] = time_end_index+1
        # remove elements after len(format)
        generated_time_trajectories[i] = generated_time_trajectories[i][:len(format)+1]

    return generated_trajectories, generated_time_trajectories