import time
from util.dataloader import *
from util.helper import *
from torch.utils.data import DataLoader
import mtnet
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import sys
import json
import pathlib

def train_epoch(epoch, gen, loader, optim):
    if epoch == 0:
    # if False:
        print("skip first epoch to evaluate the initial model")
        return float('inf'), float('inf'), float('inf'), float('inf')
    else:
        loss_list = []
        loss_x_list = []
        loss_t_list = []
        acc_loss_t_list = []
        if config.DP:
            with BatchMemoryManager(data_loader=loader, max_physical_batch_size=100, optimizer=optim) as new_data_loader:
                for (x, dpt, t, y) in new_data_loader:
                    x, dpt, t, y = x.to(config.device), dpt.to(config.device), t.to(config.device).view(-1), y.to(
                        config.device).view(-1)
                    optim.zero_grad()
                    # print(x.shape)
                    pred_x, pred_t = gen(x, dpt)
                    loss_x = -pred_x[range(pred_x.size(0)), y].sum()  # NLLLoss
                    loss_t = (pred_t[range(pred_t.size(0)), y] - t)  # [y != STOP_EDGE].abs().sum()  # MAELoss
                    loss_t[y == config.STOP_EDGE] = 0
                    loss = loss_x + config.LAMBDA * loss_t.abs().sum()  # loss_x.sum() + loss_t[y != STOP_EDGE].sum()
                    acc_loss_t_list.append(loss_t.detach().view(dpt.size()).sum(-1).abs().mean().data.item())
                    valid_nums = (y != config.STOP_EDGE).sum().data.item()
                    loss_list.append(loss.data.item() / valid_nums)
                    loss_x_list.append(loss_x.data.item() / valid_nums)
                    loss_t_list.append(loss_t.abs().sum().data.item() / valid_nums)
                    loss.backward()
                    # print gradient
                    # for p in gen.parameters():
                    #     print(p.grad_sample.shape)
                    optim.step()
        else:
            for (x, dpt, t, y) in loader:
                x, dpt, t, y = x.to(config.device), dpt.to(config.device), t.to(config.device).view(-1), y.to(
                    config.device).view(-1)
                optim.zero_grad()
                pred_x, pred_t = gen(x, dpt)
                loss_x = -pred_x[range(pred_x.size(0)), y].sum()  # NLLLoss
                loss_t = (pred_t[range(pred_t.size(0)), y] - t)  # [y != STOP_EDGE].abs().sum()  # MAELoss
                loss_t[y == config.STOP_EDGE] = 0
                loss = loss_x + config.LAMBDA * loss_t.abs().sum()  # loss_x.sum() + loss_t[y != STOP_EDGE].sum()
                acc_loss_t_list.append(loss_t.detach().view(dpt.size()).sum(-1).abs().mean().data.item())
                valid_nums = (y != config.STOP_EDGE).sum().data.item()
                loss_list.append(loss.data.item() / valid_nums)
                loss_x_list.append(loss_x.data.item() / valid_nums)
                loss_t_list.append(loss_t.abs().sum().data.item() / valid_nums)
                loss.backward()
                optim.step()
        return np.mean(loss_list), np.mean(loss_x_list), np.mean(loss_t_list), np.mean(acc_loss_t_list)


def train(name, model, loader, bm, optim):
    t0 = time.time()
    print('[**%s**] with #params: %d' % (name, get_n_params(model)), flush=True)
    # multiple gpus
    # if torch.cuda.device_count() > 1:
    #     print("Let's use %d GPUs!" % (torch.cuda.device_count()), flush=True)
    #     model = nn.DataParallel(model).to(device)
    logger = Logger(model, bm, name)
    base_epoch = -1
    # base_epoch = logger.load_history()
    for epoch in range(base_epoch + 1, config.EPOCHS):
        tepoch = time.time()
        logger.log(epoch, train_epoch(epoch, model, loader, optim), time.time() - tepoch)
        if logger.earlystop.early_stop:
            print(name, '\t', 'early stopped......')
            break
    t1 = time.time() - t0
    print('[**%s**] Done with %.1f mins\n***********************************\n' % (name, t1 / 60), flush=True)


def main():
    settings = init()
    adjs = load_edgeadjs()
    ttts, ttts_test = load_trajs_raw()
    loader = DataLoader(GenDataset(adjs, *ttts), config.BATCH_SIZE, shuffle=True, pin_memory=True)
    bm = Benchmarker(adjs, *ttts)
    name = get_full_name(settings, 'mtnet')
    gen = mtnet.Model(device=config.device)
    optim = torch.optim.Adam(gen.parameters())
    if config.DP:
        print("DP-SGD")
        privacy_engine = PrivacyEngine(accountant="prv")
        gen, optim, loader = privacy_engine.make_private(module=gen, optimizer=optim, data_loader=loader, noise_multiplier=1, max_grad_norm=1)
    train(name, gen, loader, bm, optim)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    save_dir = sys.argv[2]
    epoch = int(sys.argv[3])
    cuda_number = int(sys.argv[4])
    # convert string to bool
    if sys.argv[5] == 'True':
        config.DP = True
    else:
        config.DP = False
    config.DATA_DIR = pathlib.Path(data_dir)
    config.SAVE_DIR = pathlib.Path(save_dir)
    config.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    config.SAMPLE_SAVE_DIR = config.SAVE_DIR

    # training_data_dir=/data/${dataset}/${max_size}/${name}
    training_settings = {"dataset": data_dir.split('/')[2], "data_name": data_dir.split('/')[-2], "network_type": "MTNet"}
    with open(config.SAVE_DIR / 'params.json', 'w') as f:
        json.dump(training_settings, f)
    
    send(config.SAVE_DIR / 'params.json')
    config.PARAM_BASE = config.SAVE_DIR
    n_data = 0
    with open(config.DATA_DIR / 'training_data.csv', 'r') as f:
        for line in f:
            n_data += 1
    config.BATCH_SIZE = int(np.sqrt(n_data))
    config.EPOCHS = epoch
    config.device = torch.device(f'cuda:{cuda_number}' if torch.cuda.is_available() else 'cpu')  # + str(torch.cuda.device_count() - 1)
    main()
