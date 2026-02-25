"""
CPU-compatible runner for ClustGDD transductive training on Cora.
Patches CUDA calls to run on CPU-only environments.
"""
import torch
import sys
import os

# === CPU Patches: redirect all CUDA calls to CPU ===
torch.cuda.is_available = lambda: False
torch.cuda.set_device = lambda x: None
torch.cuda.manual_seed = lambda x: None
torch.cuda.manual_seed_all = lambda x: None
torch.cuda.max_memory_allocated = lambda *a, **k: 0

_orig_tensor_cuda = torch.Tensor.cuda
torch.Tensor.cuda = lambda self, *a, **k: self

os.chdir('/workspace/ClustGDD')
sys.path.insert(0, '/workspace/ClustGDD')

import numpy as np
import random
import time
import argparse
import torch.nn.functional as F

from utils import *
from utils_clustgdd import *
from clustgdd_agent_transduct import ClustGDD
from utils_graphsaint import DataGraphSAINT

# Use the same args as main_transduct.sh for Cora (reduction_rate=0.5 for reasonable speed)
sys.argv = [
    'run_clustgdd_cpu.py',
    '--gpu_id', '0',
    '--dataset', 'cora',
    '--reduction_rate', '0.5',
    '--prop_num', '5',
    '--postprop_num', '2',
    '--alpha', '0.8',
    '--predropout', '0.6',
    '--sp_ratio', '0.4',
    '--preep', '80',
    '--postep', '100',
    '--frcoe', '0.01',
    '--predcoe', '1.0',
    '--save', '0',
]

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=1)
parser.add_argument('--seed', type=int, default=15)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--gctype', type=str, default='clustgdd')
parser.add_argument('--prop_num', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--prehidden', type=int, default=256)
parser.add_argument('--predropout', type=float, default=0.6)
parser.add_argument('--prewd', type=float, default=5e-4)
parser.add_argument('--prelr', type=float, default=0.01)
parser.add_argument('--preep', type=int, default=600)
parser.add_argument('--prenlayers', type=int, default=2)
parser.add_argument('--cluster_minibatch', type=int, default=1000)
parser.add_argument('--sp_ratio', type=float, default=0.05)
parser.add_argument('--sp_type', type=str, default='attaw')
parser.add_argument('--postep', type=int, default=100)
parser.add_argument('--postprop_num', type=int, default=1)
parser.add_argument('--postlr_feat', type=float, default=1e-4)
parser.add_argument('--postlr_adj', type=float, default=1e-4)
parser.add_argument('--postlr_model', type=float, default=1e-2)
parser.add_argument('--postwd_feat', type=float, default=5e-4)
parser.add_argument('--postwd_adj', type=float, default=5e-4)
parser.add_argument('--postwd_model', type=float, default=5e-4)
parser.add_argument('--frcoe', type=float, default=0.01)
parser.add_argument('--predcoe', type=float, default=1.)
parser.add_argument('--csttemp', type=float, default=0.5)
parser.add_argument('--w1', type=float, default=0.1)
parser.add_argument('--w2', type=float, default=1.)
parser.add_argument('--no_refinement', type=bool, default=False)
parser.add_argument('--no_adjsyn', type=bool, default=False)
parser.add_argument('--save_pretrained_output', type=bool, default=False)
parser.add_argument('--save_syn_output', type=bool, default=False)
parser.add_argument('--save_norf', type=bool, default=False)
parser.add_argument('--notopo', type=bool, default=False)
parser.add_argument('--tm_rec', type=bool, default=False)

args = parser.parse_args()

# CPU device instead of CUDA
device = 'cpu'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("=" * 60)
print("ClustGDD Transductive Training on Cora (CPU)")
print("=" * 60)
print(args)

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

print(f"\nDataset: {args.dataset}")
print(f"Train nodes: {len(data.idx_train)}, Val nodes: {len(data.idx_val)}, Test nodes: {len(data.idx_test)}")
print(f"Features dim: {data.feat_train.shape[1]}, Classes: {data.nclass}")

if args.gctype == 'clustgdd':
    agent = ClustGDD(data, args, device=device)

print("\nStarting training...")
t_start = time.time()
agent.train()
t_end = time.time()

print(f"\n{'=' * 60}")
print(f"Training completed in {t_end - t_start:.2f} seconds")
print(f"{'=' * 60}")
