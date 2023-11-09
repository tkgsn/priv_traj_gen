import math
import numbers
import numpy as np
import torch
from torch.utils.data import Dataset
from util import config
import random
import os


def normalization(x, dim=-1, eps: numbers.Real = 1e-5):
    mean = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdim=True)

    return (x - mean) / (std + eps)


# Load from raw traj file with different lens
def load_trajs_raw():
    trajs, tdpts, tcosts = [], [], []  # T+1, T, T
    lens = []
    size_cnt = 0
    data_name = 'training_data.csv'
    time_data_name = 'training_data_time.csv'
    # data_name = 'trajs_demo.csv'
    # time_data_name = 'tstamps_demo.csv'
    # with open(os.path.join(config.SAVE_DIR, data_name), "r") as ftraj:
        # with open(os.path.join(config.SAVE_DIR, time_data_name), 'r') as ft:  # read trajs and tstamps
    with open(os.path.join(config.DATA_DIR, data_name), "r") as ftraj:
        with open(os.path.join(config.DATA_DIR, time_data_name), 'r') as ft:  # read trajs and tstamps
            for line_traj, line_t in zip(ftraj, ft):
                traj = [int(x) for x in line_traj.split()]
                if config.TRIM_STOP: traj = traj[:-1]  # trim last stop_edge flag
                if config.FIX_LEN and len(traj) > config.TRAJ_FIX_LEN:
                    traj = traj[:config.TRAJ_FIX_LEN]
                lens.append(len(traj))
                traj += [0 for _ in range(len(traj), config.TRAJ_FIX_LEN)]

                tdpt = [float(x) for x in line_t.split()]
                if config.FIX_LEN and len(tdpt) > config.TRAJ_FIX_LEN + 1:
                    tdpt = tdpt[:config.TRAJ_FIX_LEN + 1]
                tdpt += [0 for _ in range(len(tdpt[1:]), config.TRAJ_FIX_LEN)]
                tcost = tdpt[1:]  # T+1
                # if any([True if t >= 6000 else False for t in tcost]): continue
                if any(map(math.isnan, tdpt)): continue  # print(tdpt)
                if any(map(math.isnan, tcost)): continue  # print(tcost)
                tdpts.append(tdpt)  # T+2
                tcosts.append(tcost)  # T+1
                trajs.append(traj)  # T+1
                # size_cnt += 1
                # if (size_cnt >= 100000): break
                # used_num = MAX_USE_TRAJS_NUM if len(trajs) > MAX_USE_TRAJS_NUM else len(trajs)
    
    trajs, tdpts, tcosts = torch.LongTensor(trajs), torch.LongTensor(tdpts), torch.FloatTensor(tcosts)
    tdpts[:, 0] = (tdpts[:, 0] + 8 * 3600) % config.T_LOOP  # 8 hours offset
    for i in range(1, config.TRAJ_FIX_LEN + 1):
        tdpts[:, i] = ((tdpts[:, i - 1] + tdpts[:, i]) % config.T_LOOP)
    config.TCOST_MAX = tcosts.max().detach().item()
    tdpts, tcosts = tdpts[:, 1:-1], tcosts[:, 1:] / config.TCOST_MAX  # skip the one of beginning edge
    assert tdpts.size() == tcosts.size() and trajs[:, 1:].size() == tdpts.size()
    split_point = int(len(trajs) * config.SPLIT_PER)  # 70% to 30%, training and test
    used_training = int(config.USE_PER * split_point)
    shuffle_idx = list(range(len(trajs)))
    # random.shuffle(shuffle_idx)
    trajs, tdpts, tcosts = trajs[shuffle_idx], tdpts[shuffle_idx], tcosts[shuffle_idx]
    # write trajs[:used_training] to file
    with open(os.path.join(config.SAVE_DIR / 'original_trajs.csv'), 'w') as f:
        for traj in trajs[:used_training]:
            f.write(','.join([str(x) for x in traj.tolist()]) + '\n')

    print('%s loaded, TL=%d\t#trajs=%d (Tr=%d,Te=%d)\tPeriod=%ds\tMAX_COST=%d\tAVG_TCOST=%.1f\tAVG_LEN=%.1f' % (
        config.SAVE_DIR, config.TRAJ_FIX_LEN, trajs.shape[0], used_training, trajs.shape[0] - split_point, config.T_LOOP,
        config.TCOST_MAX, tcosts.sum(-1).mean().item() * config.TCOST_MAX, np.array(lens).mean()), flush=True)
    return (trajs[:used_training], tdpts[:used_training], tcosts[:used_training]), \
           (trajs[split_point:], tdpts[split_point:], tcosts[split_point:])


def load_eproperty(pname, dtype=float):
    edge_property = [dtype(0)]
    with open(os.path.join(config.DATA_DIR / 'edge_property.txt')) as f:
        for line in f:
            line = line.strip().split(',', maxsplit=4)  # edge_idx,road_len,road_type,heading,WKT
            edge_property.append(dtype(line[config.PIDX[pname]]))
    # print(pname, edge_property[:3], flush=True)
    return edge_property


def load_edgeproperties():
    edge_property = [[0 for _ in range(config.PROPERTY_DIM)]]
    roadtype_emb = np.eye(config.ROAD_TYPES)
    with open(os.path.join(config.DATA_DIR / 'edge_property.txt')) as f:
        for line in f:
            line = line.split(',')  # edge_idx,road_len,road_type,heading,WKT
            rad = float(line[config.PIDX['heading']]) / 180. * math.pi - math.pi  # [-pi,pi]
            edge_property.append([*roadtype_emb[int(line[config.PIDX['road_type']])],
                                  float(line[config.PIDX['len']]), rad, math.sin(rad)])
    edge_property += [[0 for _ in range(config.PROPERTY_DIM)]]  # not used mask
    edge_property = torch.Tensor(edge_property)
    # normalize road length, headings.
    edge_property[:, config.ROAD_TYPES:] = normalization(edge_property[:, config.ROAD_TYPES:], dim=-1)
    # print('edge property: ', edge_property[:3], flush=True)
    return edge_property


def load_edgeadjs():  # default mask is -1
    adjs = []
    with open(os.path.join(config.DATA_DIR / 'edge_adj.txt'), 'r') as f:
        for line in f:
            adjs.append([int(e) for e in line.strip(',').split(',')])
            # if adjs[-1][0] == -1: adjs[-1][0] = STOP_EDGE  # dead road
        adjs = [[-1 for _ in range(len(adjs[0]))]] + adjs + [[-1 for _ in range(len(adjs[0]))]]
    adjs = torch.cat([torch.LongTensor(adjs), torch.zeros(len(adjs), 1, dtype=torch.long)], dim=-1)  # zero is STOP_EDGE
    adjs[adjs == -1] = adjs.size(0) - 1  # change mask to size-1, -1 can not look up embedding
    # print('adjs', adjs[:2], '\n', adjs[-1])
    return adjs


def load_time_emb():
    init_values = torch.FloatTensor(7 * 48, 32)
    with open(os.path.join(config.DATA_BASE + config.CITY, 'temporal_32.emb'), 'r') as f:
        rows, emb_size = f.readline().split()
        assert int(rows) == 7 * 48 and int(emb_size) == 32
        for line in f:
            line = line.split()
            idx, emb_vec = int(line[0]), torch.Tensor(list(map(float, line[1:])))
            init_values[idx] = emb_vec
        # print('#tunits: ', rows, '\tt_emb_size: ', emb_size)
        return init_values


# dataset for pre train generator
class GenDataset(Dataset):
    def __init__(self, adjs, trajs, tdpts, tcosts):
        assert trajs[:, 1:].size() == tdpts.size() and tdpts.size() == tcosts.size()
        self.trajs = trajs
        # self.dpt_slots = tdpts / config.TRANGE_BASE
        # convert dpt to int
        # self.dpt_slots = (tdpts / config.TRANGE_BASE).int()
        self.dpt_slots = process_tdpts(tdpts)
        self.tcosts = tcosts
        self.adjs = adjs

    def __len__(self):
        return self.trajs.size(0)

    def __getitem__(self, index):
        traj = self.trajs[index]
        y_idx_flat = (self.adjs[traj[:-1]] == traj[1:].unsqueeze(-1)).nonzero()[:, -1]  # B*L
        # print(traj)
        # for i, state in enumerate(traj[:-1]):
            # print(state, "->", traj[i+1], self.adjs[state])
        assert traj.shape[0]-1 == y_idx_flat.shape[0], 'traj shape: %s, y_idx_flat shape: %s' % (traj.shape, y_idx_flat.shape)
        return (traj[:-1].long(), self.dpt_slots[index].long(), self.tcosts[index], y_idx_flat)

def process_tdpts(tdpts):
    dpt_slots = (tdpts / config.TRANGE_BASE).int()
    return dpt_slots