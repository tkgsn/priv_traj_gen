import torch.nn as nn
from util import dataloader as loader
from util.dataloader import process_tdpts
import torch.nn.functional as F
from util import config
import torch
from opacus.layers.dp_rnn import DPLSTMCell as LSTMCell

class Model(nn.Module):
    def __init__(self, z_dim=16, t_dim=48, te_dim=32, xe_dim=128, h_dim=128, o_dim=32, device=None):
        super(Model, self).__init__()
        self.z_dim = z_dim
        self.register_buffer('adj', loader.load_edgeadjs())
        self.register_buffer('eprop', loader.load_edgeproperties())
        if config.USE_TGRAPH:
            self.t_emb = nn.Embedding(t_dim * 7, te_dim)
            # self.t_emb = nn.Linear(t_dim * 7, te_dim)
            self.t_emb.weight = nn.Parameter(loader.load_time_emb(), True)  # load pre-trained temporal graph values
            self.tcost_emb = nn.Embedding(self.adj.size(0), te_dim)  # New
        else:
            self.t_emb = nn.Embedding(t_dim, te_dim)
            self.tcost_emb = nn.Embedding(self.adj.size(0), te_dim)  # New

        self.x_emb = nn.Embedding(self.adj.size(0), xe_dim)  # embed edges, 0 for padding

        self.lstms = nn.ModuleList()  # nn.LSTMCell(h_dim, h_dim) for _ in range(LSTMS_NUM)
        
        bias = True
        self.lstms.append(LSTMCell(z_dim + te_dim + xe_dim + self.eprop.size(1), h_dim, bias))
        for _ in range(1, config.LSTMS_NUM - 1):
            self.lstms.append(LSTMCell(h_dim, h_dim, bias))
        self.lstms.append(LSTMCell(h_dim, o_dim, bias)) if config.L2_NUM == 0 else self.lstms.append(
            LSTMCell(h_dim, h_dim, bias))

        self.linear_afterx = []  # nn.Sequential()
        self.linear_aftert = []
        self.linear_aftert += [nn.Linear(te_dim + te_dim, o_dim), nn.ReLU()]
        if config.L2_NUM != 0:
            self.linear_afterx += [nn.Linear(h_dim, o_dim), nn.ReLU()]
        for _ in range(1, config.L2_NUM):
            self.linear_afterx += [nn.Linear(o_dim, o_dim), nn.ReLU()]
            self.linear_aftert += [nn.Linear(o_dim, o_dim), nn.ReLU()]
        self.linear_afterx = nn.Sequential(*self.linear_afterx)
        self.linear_aftert = nn.Sequential(*self.linear_aftert)
        self.mlp_meta_x = nn.Sequential(nn.Linear(xe_dim + te_dim + self.eprop.size(1), o_dim))
        self.mlp_meta_t = nn.Sequential(nn.Linear(xe_dim + te_dim + self.eprop.size(1), o_dim))
        self.to(device)

    def adj_idx(self, input, output):  # return adj idx
        return (self.adj[input] == output.unsqueeze(-1)).nonzero()[:, -1]  # (B*L, )

    def init_h(self, batch_size, device):
        hs, cs = [], []
        for lstm in self.lstms:
            hs.append(torch.zeros(batch_size, lstm.hidden_size, device=device))
            cs.append(torch.zeros(batch_size, lstm.hidden_size, device=device))
        return hs, cs

    def forward(self, x, dpt):  # B * L, traj sequences
        x, dpt = x.t(), dpt.t()
        L, B = x.size()
        pred_x, pred_t = [], []
        hs, cs = self.init_h(B, x.device)
        for i in range(L):
            hidden, hs, cs = self.step(x[i], dpt[i], hs, cs)
            pred_x.append(F.log_softmax(hidden[0], dim=-1))  # B*adj_dim
            pred_t.append(torch.sigmoid(hidden[1]))

        pred_x = torch.stack(pred_x, 1).view(-1, self.adj.size(1))  # (B*L) * adj_dim
        pred_t = torch.stack(pred_t, 1).view(-1, self.adj.size(1))
        return pred_x, pred_t

    def step(self, xt, dptslot, hs, cs):  # B
        # print(xt.shape)
        B = xt.size(0)
        zt = torch.randn(B, self.z_dim, device=xt.device)
        xt_emb = self.x_emb(xt)  # B * emb_dim
        # print(dptslot.max(), dptslot.min())
        tt_emb = self.t_emb(dptslot)  # B * temb_dim
        xt_eprop = self.eprop[xt]  # B * eprop_dim
        hidden = torch.cat([zt, xt_emb, tt_emb, xt_eprop], dim=-1)
        for i, lstm in enumerate(self.lstms):  # LSTMs
            hs[i], cs[i] = lstm(hidden, (hs[i], cs[i]))
            hidden = hs[i]
        afterx = self.linear_afterx(hidden)
        adjs = self.adj[xt]  # B * adj_dim
        tt_emb = tt_emb.unsqueeze(1).expand(-1, self.adj.size(1), -1)
        aftert = self.linear_aftert(torch.cat([self.tcost_emb(adjs), tt_emb], dim=-1))
        props = self.eprop[adjs]  # .view(B, self.adj_dim, -1)
        mk = torch.cat([self.x_emb(adjs), tt_emb, props], -1)  # B * adj_size * know_emb_len
        softmax_w = self.mlp_meta_x(mk)
        sigmoid_w = self.mlp_meta_t(mk)

        pred_xnext = afterx.view(B, 1, 1, -1).matmul(softmax_w.view(B, self.adj.size(1), -1, 1)).view(B, -1)  # B * adj_dim
        pred_xnext[adjs == self.adj.size(0) + config.ADJ_MASK] = float('-inf')
        pred_tnext = aftert.unsqueeze(2).matmul(sigmoid_w.unsqueeze(-1)).view(B, self.adj.size(1))
        return (pred_xnext, pred_tnext), hs, cs

    def sample(self, x, dpts, ts):  # B * L, parts of sequences
        with torch.no_grad():
            if x is None or len(x) == 0: return torch.LongTensor().to(config.device), torch.FloatTensor().to(config.device)
            # assert x is not None and len(x) > 0
            x, dpts, ts = x.t(), dpts.t(), ts.t()
            given_len, B = x.size()
            samples_x, samples_t = [], []
            hs, cs = self.init_h(B, x.device)
            for i in range(given_len):
                hidden, hs, cs = self.step(x[i], dpts[i], hs, cs)
                if i + 1 < given_len: samples_x.append(x[i + 1]), samples_t.append(ts[i])
            sample_batch_idx = F.softmax(hidden[0], -1).multinomial(1).view(-1)  # .argmax(-1)
            samples_x.append(self.adj[x[-1]][range(B), sample_batch_idx])
            samples_t.append(torch.sigmoid(hidden[1][range(B), sample_batch_idx]))
            dpt = (dpts[-1] + (samples_t[-1] * config.TCOST_MAX).round().long()) % config.T_LOOP
            dpt = process_tdpts(dpt)
            for i in range(given_len, config.TRAJ_FIX_LEN - 1):
                hidden, hs, cs = self.step(samples_x[-1], dpt, hs, cs)
                sample_batch_idx = F.softmax(hidden[0], -1).multinomial(1).view(-1)  # .multinomial(1).view(-1)
                samples_x.append(self.adj[samples_x[-1]][range(B), sample_batch_idx])
                samples_t.append(torch.sigmoid(hidden[1][range(B), sample_batch_idx]))
                dpt = (dpt + (samples_t[-1] * config.TCOST_MAX).round().long()) % config.T_LOOP
                dpt = process_tdpts(dpt)
            samples_x, samples_t = torch.stack(samples_x, 1), torch.stack(samples_t, 1)  # B*L
            return samples_x, samples_t

    def sample_t(self, x, yidx, dpts):
        with torch.no_grad():
            if x is None or len(x) == 0: return torch.FloatTensor().to(config.device)
            x, yidx, dpts_slots = x.t(), yidx.t(), dpts.t()
            L, B = x.size()
            assert L == config.TRAJ_FIX_LEN - 1
            samples_costs = []
            hs, cs = self.init_h(B, x.device)
            for i in range(L):
                hidden, hs, cs = self.step(x[i], dpts_slots[i], hs, cs)
                pred_t = torch.sigmoid(hidden[1][range(B), yidx[i]])
                pred_t[yidx[i] == config.STOP_EDGE] = 0
                samples_costs.append(pred_t)
            return torch.stack(samples_costs, 1)  # B*L
