import torch
import pathlib
import numpy as np

# NAME="geolife"
# DATA_DIR = pathlib.Path("/data/geolife/0/narrow_0_0_bin30_seed0")
# SAVE_DIR = pathlib.Path("/data/results/geolife/0/narrow_0_0_bin30_seed0/DP_MTNet")
# SAMPLE_SAVE_DIR = pathlib.Path("/data/results/geolife/0/narrow_0_0_bin30_seed0/DP_MTNet")
# PARAM_BASE = SAVE_DIR
# SAVE_DIR.mkdir(parents=True, exist_ok=True)


NAME=""
DATA_DIR = ""
SAVE_DIR = ""
SAMPLE_SAVE_DIR = ""
PARAM_BASE = ""

RES_FILE = 'res.csv'

# count the number of data in ./data/geolife/trajs_demo.csv
n_data = 0
max_len = 0
# with open(DATA_DIR / 'training_data.csv', 'r') as f:
    # for line in f:
        # n_data += 1
        # max_len = max(max_len, len(line.split()))
# BATCH_SIZE = int(np.sqrt(n_data))
BATCH_SIZE = 0

EPOCHS = 50
# print("max length", max_len)
TRAJ_FIX_LEN = 20

THRESHOLD = 5

LAMBDA = 0.1

LSTMS_NUM = 2
L2_NUM = 1

# device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')  # + str(torch.cuda.device_count() - 1)

USE_PER = 1
# SPLIT_PER = 1/30
# SPLIT_PER = 0.8
SPLIT_PER = 1
GENE_RATIO = 1


USE_TGRAPH = False
TRANGE_BASE = 1800  # half an hour
T_ONEDAY = 3600 * 24
T_LOOP = T_ONEDAY if not USE_TGRAPH else T_ONEDAY * 7  # consider one day loop or one week

TCOST_MAX = 6000

FIX_LEN = True
TRIM_STOP = True  # true if extra 0 along traj line


EPSILON = 1e-3

DP = True

ROAD_TYPES = 7
HEADING_DIM = 2
ROAD_LEN_DIM = 1
PROPERTY_DIM = ROAD_TYPES + HEADING_DIM + ROAD_LEN_DIM

ADJ_MASK = -1
STOP_EDGE = 0

PIDX = {'id': 0, 'len': 1, 'road_type': 2, 'heading': 3, 'WKT': 4}

# print('Current device: ', device, flush=True)