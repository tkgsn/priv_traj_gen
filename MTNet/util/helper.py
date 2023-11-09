from matplotlib import pyplot as plt
from util.dataloader import *
import random
from statistics import mean
import datetime
from os import path
from . import config
from .earlystop import EarlyStopping
import yaml
import subprocess


def send(path):

    source_file_path = path
    destination_file_path = f'evaluation-server:{path.parent}'

    print('ssh', 'evaluation-server', f"'mkdir -p {path.parent}'")
    print('scp', source_file_path, destination_file_path)
    result = subprocess.run(['ssh', '-o', 'StrictHostKeyChecking=no', 'evaluation-server', f"mkdir -p {path.parent}"])
    result = subprocess.run(['scp', '-o', 'StrictHostKeyChecking=no', source_file_path, destination_file_path])

def tstrip(traj):
    ridx = len(traj) - 1
    while traj[ridx] == 0: ridx -= 1
    return traj[:ridx + 1]


def draw(names, xs, ys):
    for name, x, y in zip(names, xs, ys):
        plt.plot(x, y)
        plt.savefig(name)
        plt.close()
        

def jsd(d1, d2):
    divergence = 0
    for k, p1 in d1.items():
        p2 = 0 if k not in d2 else d2[k]
        divergence += p1 * np.log2(2 * p1 / (p1 + p2))
    for k, p2 in d2.items():
        p1 = 0 if k not in d1 else d1[k]
        divergence += p2 * np.log2(2 * p2 / (p1 + p2))
    return divergence / 2


class Logger:
    def __init__(self, model, bm, model_name):
        self.name = model_name
        if config.DP:
            self.model = model._module
        else:
            self.model = model
        self.bm = bm
        self.earlystop = EarlyStopping(path=str(config.PARAM_BASE) + "/" + model_name + '_best.pth')
        self.watches = ['loss', 'loss_x', 'loss_t', 'acc_t', 'eval_des', 'eval_route', 'eval_t']
        for attr in self.watches: setattr(self, attr, [])
        self.epoch = 0

    def log(self, epoch, losses, duration=0.):
        self.epoch = epoch
        for i, loss in enumerate(self.watches[:4]): getattr(self, loss).append(losses[i])
        evals = self.bm.eval(self.model, epoch)
        for i, eval in enumerate(self.watches[4:]): getattr(self, eval).append(evals[i])
        # self.earlystop(evals[1], self.model)
        self.earlystop(losses[1], self.model)
        print('Epoch[%d] = %.1fmins\tL = %.2f\tL[x] = %.2f,\tL[t] = %.2f\tE[des] = %.4f\tE[route] = %.4f\tE[t] = %.2f' %
              (epoch, duration / 60., losses[0], losses[1], losses[2] * config.TCOST_MAX, evals[0], evals[1], evals[2]),
              flush=True)
        self.save_history(epoch)

    def load_history(self):
        try:
            check_point = torch.load(config.PARAM_BASE / (self.name + '.pth'), map_location=config.device)
            self.epoch = check_point['epoch']
            try:
                self.model.load_state_dict(check_point['state'])
            except:  # from multiple GPU to single or none
                self.model.load_state_dict({k.partition('module.')[2]: v for (k, v) in check_point['state'].items()})
            self.model.to(config.device)
            for attr in self.watches: setattr(self, attr, check_point[attr])
            print('Resume %s,epoch %d' % (self.name, self.epoch), flush=True)
            return self.epoch
        except Exception as e:
            print(e, ' Initialize model for %s' % self.name, flush=True)
            return -1

    def save_history(self, epoch):
        attrs = {attr: getattr(self, attr) for attr in self.watches}
        attrs['epoch'] = epoch
        attrs['state'] = self.model.state_dict()
        torch.save(attrs, config.PARAM_BASE / (self.name + '.pth'))
        # if epoch == config.EPOCHS - 1 or self.earlystop.early_stop:  # dump result, error for one eval,from 100->200
        #     if not path.exists(config.DATA_BASE + config.CITY + config.RES_FILE):
        #         with open(config.DATA_BASE + config.CITY + config.RES_FILE, 'w') as f:
        #             head = 'name,epoch'
        #             for attr in self.watches: head += ',' + attr
        #             f.write(head + '\n')
        #     with open(config.DATA_BASE + config.CITY + config.RES_FILE, 'a+') as f:
        #         line = self.name + ',' + str(epoch)
        #         for attr in self.watches: line += ',' + str(getattr(self, attr)[-1])
        #         f.write(line + '\n')
        #         print('finish training: ', self.name)


class Benchmarker:
    def __init__(self, adjs, trajs, tdpts, tcosts, file='util/imgs/samples.csv'):
        self.adjs = adjs
        self.WKTs = load_eproperty('WKT', str)
        # self.lens = load_eproperty('len')
        # self.trajs, self.tdpts, self.tcosts = trajs, tdpts, tcosts  # trajs for test
        self.trajs, self.tdpts, self.tcosts = trajs, process_tdpts(tdpts), tcosts  # trajs for test
        self.src_num = [0 for _ in range(len(self.adjs))]
        self.src_tdpts = [[] for _ in range(len(self.adjs))]

        for i, traj in enumerate(self.trajs.tolist()):  # collect cnt and dpt stats
            traj = tstrip(traj)
            src, des = (traj[0], traj[-1])
            self.src_num[src] += 1
            self.src_tdpts[src].append(self.tdpts[i][0].data.item())
        # self.src_valid = [cnt >= config.THRESHOLD for cnt in self.src_num]
        assert self.src_num[0] == 0 and self.src_num[-1] == 0
        self.des_density, self.route_density, self.src_des_num = self.proc_density(trajs.tolist())
        # self.src_des_valid = {(s, d): v >= config.SRC_DES_THRESHOLD for (s, d), v in self.src_des_num.items()}
        self.cnt = 0
        # describe the distribution of evaluation data
        szs_origin = []
        for i in range(len(self.adjs)):
            if len(self.des_density[i]) == 0: continue
            szs_origin.append(len(self.des_density[i]))
        # print(szs_origin)
        print('Evalution data stats -- fixed origin: min %d, mean %d, 20%% %d, median %d, max %d' % (
            np.min(szs_origin), np.mean(szs_origin), np.percentile(szs_origin, 20), np.median(szs_origin),
            np.max(szs_origin)), flush=True)

    def proc_density(self, trajs):  # process for des and route density
        des_density = [{} for _ in range(len(self.adjs))]  # des density for each (src,des) pair, src fixed
        route_density = [{} for _ in range(len(self.adjs))]  # edge usage density for each edge, src fixed
        src_des_num = {}
        src_num = [0 for _ in range(len(self.adjs))]
        for i, traj in enumerate(trajs):  # iter all trajs
            traj = tstrip(traj)
            src, des = (traj[0], traj[-1])
            src_num[src] += 1
            src_des_num[(src, des)] = 1. if (src, des) not in src_des_num else src_des_num[(src, des)] + 1
            des_density[src][des] = 1. if des not in des_density[src] else des_density[src][des] + 1.0
            if len(traj) <= 2: continue
            e_w = 1.0  # / (len(traj) - 2) <- result in each dim distribution (binary)
            # e_w = 1.0 / (len(traj) -2) <- result in distribution on all locations
            for e in traj[1:-1]:  # share by the whole traj
                route_density[src][e] = e_w if e not in route_density[src] else route_density[src][e] + e_w
        for e in range(1, len(self.adjs) - 1):  # transfer to probs.
            des_density[e] = {k: v / src_num[e] for k, v in des_density[e].items()}
            route_density[e] = {k: v / src_num[e] for k, v in route_density[e].items()}
        return des_density, route_density, src_des_num

    def eval_density(self, des_density, route_density, src_des_num):
        dist_des, dist_route = 0.0, 0.0
        src_num = 0
        for eid in range(len(self.adjs)):
            if self.src_num[eid] < config.THRESHOLD: continue
            dist_des += jsd(des_density[eid], self.des_density[eid])
            dist_route += jsd(route_density[eid], self.route_density[eid])
            src_num += 1.
        if src_num > 0:
            dist_des /= src_num
            dist_route /= src_num
        return dist_des, dist_route

    def eval(self, gen, epoch):
        samples = []
        samples_t = []
        mae_cost_acc = 0.
        costb = torch.FloatTensor(config.BATCH_SIZE, 0).to(config.device)  # empty place holder
        # for i in range(len(self.trajs) // config.BATCH_SIZE + 1):
        #     lr, rr = i * config.BATCH_SIZE, (i + 1) * config.BATCH_SIZE
        #     if rr > len(self.trajs): rr = len(self.trajs)
        #     # for traj route
        #     tb, dptb = self.trajs[lr:rr, :1].to(config.device), self.tdpts[lr:rr, :1].to(config.device)
        #     # print("a", dptb.max())
        #     samples_batch, _ = gen.sample(tb, dptb, costb) if not hasattr(gen, 'module') else gen.module.sample(tb, dptb, costb)
        #     samples += (torch.cat((tb, samples_batch), 1)).tolist()  # append the fixed source
        #     # for traj cost
        #     yidxb = (self.adjs[self.trajs[lr:rr, :-1]] == self.trajs[lr:rr, 1:].unsqueeze(-1)).nonzero()[:,
        #             -1].reshape(rr - lr, -1)
        #     tb, yidxb, dptb = self.trajs[lr:rr, :-1].to(config.device), yidxb.to(config.device), self.tdpts[lr:rr].to(
        #         config.device)
        #     sample_costb = gen.sample_t(tb, yidxb, dptb) if not hasattr(gen, 'module') else gen.module.sample_t(
        #         tb, yidxb, dptb)
        #     mae_cost_acc += (sample_costb - self.tcosts[lr:rr].to(config.device)).abs().sum().item() / (
        #             self.trajs[lr:rr, 1:] != config.STOP_EDGE).sum().item()

        # change to get arbitrary number of samples
        n_generated = len(self.trajs) * config.GENE_RATIO
        batch_for_generation = 100
        for _ in range(n_generated // batch_for_generation):
            indice = np.random.choice(len(self.trajs), batch_for_generation)
            # for traj route
            # tb, dptb = self.trajs[lr:rr, :1].to(config.device), self.tdpts[lr:rr, :1].to(config.device)
            tb, dptb = self.trajs[indice, :1].to(config.device), self.tdpts[indice, :1].to(config.device)
            # print("a", dptb.max())
            samples_batch, _ = gen.sample(tb, dptb, costb) if not hasattr(gen, 'module') else gen.module.sample(tb, dptb, costb)
            samples += (torch.cat((tb, samples_batch), 1)).tolist()  # append the fixed source
            # for traj cost
            # yidxb = (self.adjs[self.trajs[lr:rr, :-1]] == self.trajs[lr:rr, 1:].unsqueeze(-1)).nonzero()[:,-1].reshape(rr - lr, -1)
            yidxb = (self.adjs[self.trajs[indice, :-1]] == self.trajs[indice, 1:].unsqueeze(-1)).nonzero()[:,-1].reshape(batch_for_generation, -1)
            # print("yidxb", yidxb)
            # tb, yidxb, dptb = self.trajs[lr:rr, :-1].to(config.device), yidxb.to(config.device), self.tdpts[lr:rr].to(config.device)
            tb, yidxb, dptb = self.trajs[indice, :-1].to(config.device), yidxb.to(config.device), self.tdpts[indice].to(config.device)
            sample_costb = gen.sample_t(tb, yidxb, dptb) if not hasattr(gen, 'module') else gen.module.sample_t(tb, yidxb, dptb)

            samples_t += (torch.cat((torch.zeros(sample_costb.shape[0], 1, dtype=int).to(config.device), (sample_costb*config.TCOST_MAX).int()), 1)).tolist()

            # mae_cost_acc += (sample_costb - self.tcosts[lr:rr].to(config.device)).abs().sum().item() / (self.trajs[lr:rr, 1:] != config.STOP_EDGE).sum().item()
            mae_cost_acc += (sample_costb - self.tcosts[indice].to(config.device)).abs().sum().item() / (self.trajs[indice, 1:] != config.STOP_EDGE).sum().item()

        save_dir = config.SAMPLE_SAVE_DIR / f"model_{epoch}"
        (config.SAMPLE_SAVE_DIR / f"model_{epoch}").mkdir(parents=True, exist_ok=True)
        # write samples
        with open(save_dir / f"samples.txt", 'w') as f:
            for sample in samples:
                f.write(','.join(list(map(str, sample))) + '\n')    
        # write samples_t
        with open(save_dir / f"samples_time.txt", 'w') as f:
            for sample in samples_t:
                f.write(','.join(list(map(str, sample))) + '\n')
        
        print(len(samples), "data generated to", save_dir / f"samples.txt")
        print(len(samples_t), "time data generated to", save_dir / f"samples_time.txt")

        send(save_dir / f"samples.txt")
        send(save_dir / f"samples_time.txt")

        # assert len(samples) == len(self.trajs)
        dist_des, dist_route = self.eval_density(*self.proc_density(samples))
        mae_cost_acc = mae_cost_acc / (len(self.trajs) // config.BATCH_SIZE + 1) * config.TCOST_MAX

        return dist_des, dist_route, mae_cost_acc


def init():
    file = 'util/mtnet.yaml'
    # read models configuration
    with open(file, 'r') as f:
        settings = yaml.load(f, yaml.CLoader)
    assert file[:-5].endswith(settings['name'])

    # seed random
    torch.manual_seed(settings['seed'])
    np.random.seed(settings['seed'])
    random.seed(settings['seed'])

    return settings


def get_n_params(model):
    try:
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn *= s
            pp += nn
        return pp
    except Exception as e:
        # print('Get params num in multiGPUs mode ', e)
        pp = 0
        for p in list(model.module.parameters()):
            nn = 1
            for s in list(p.size()):
                nn *= s
            pp += nn
        return pp


def get_full_name(settings, model):
    model_name = config.NAME + '\t' + model + '\tB:' + str(
        config.BATCH_SIZE) + '\tused' + str(config.USE_PER * 100) + '%' + '\tTL:' + str(
        config.TRAJ_FIX_LEN - 1) + '\tLMDA:' + str(config.LAMBDA) + ' #RNN:' + str(
        config.LSTMS_NUM) + ' #L2:' + str(config.L2_NUM)
    hyper_params = ['z_dim', 't_dim', 'te_dim', 'xe_dim', 'h_dim', 'o_dim']
    for par in hyper_params:
        model_name += ' ' + par[:-4] + ':' + str(settings[par])
    return model_name