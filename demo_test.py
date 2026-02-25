"""
Demo script to verify the development environment for both ClustGDD and Rankformer.
Runs on CPU (no GPU required).
"""
import sys
import os

print("=" * 60)
print("Development Environment Verification Demo")
print("=" * 60)

# ===== Part 1: Verify all imports =====
print("\n[1/4] Verifying all package imports...")
import torch
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
import torch_geometric
import torch_scatter
import torch_sparse
import torch_cluster
import ogb
import networkx
import deeprobust
print(f"  PyTorch:       {torch.__version__}")
print(f"  NumPy:         {np.__version__}")
print(f"  Pandas:        {pd.__version__}")
print(f"  SciPy:         {scipy.__version__}")
print(f"  Scikit-learn:  {sklearn.__version__}")
print(f"  Matplotlib:    {matplotlib.__version__}")
print(f"  PyG:           {torch_geometric.__version__}")
print(f"  OGB:           {ogb.__version__}")
print(f"  NetworkX:      {networkx.__version__}")
print("  All imports OK!")

# ===== Part 2: Test Rankformer on CPU =====
print("\n[2/4] Testing Rankformer (Graph Transformer for Recommendation)...")
sys.path.insert(0, '/workspace/Rankformer/code')

sys.argv = [
    'demo', '--data=TestData', '--use_gcn', '--use_rankformer',
    '--rankformer_layers=2', '--max_epochs=3', '--valid_interval=1',
    '--device=0', '--seed=42'
]

os.environ['CUDA_VISIBLE_DEVICES'] = ''
orig_cuda_available = torch.cuda.is_available
torch.cuda.is_available = lambda: False

from parse import args
args.device = torch.device('cpu')
args.data_dir = '/workspace/Rankformer/data/'
args.train_file = os.path.join(args.data_dir, args.data, 'train.txt')
args.valid_file = os.path.join(args.data_dir, args.data, 'valid.txt')
args.test_file = os.path.join(args.data_dir, args.data, 'test.txt')

from dataloader import MyDataset
dataset = MyDataset(args.train_file, args.valid_file, args.test_file, args.device)
print(f"  Dataset loaded: {dataset.num_users} users, {dataset.num_items} items")

from model import Model
model = Model(dataset).to(args.device)
print(f"  Model created: {sum(p.numel() for p in model.parameters())} parameters")

for epoch in range(1, 4):
    loss = model.train_func()
    print(f"  Epoch {epoch}: train_loss = {loss.item():.6f}")

valid_pre, valid_recall, valid_ndcg = model.valid_func()
print(f"  Validation NDCG@20: {valid_ndcg[0]:.6f}")
print("  Rankformer demo PASSED!")

# ===== Part 3: Test ClustGDD imports =====
print("\n[3/4] Testing ClustGDD (Graph Distillation via Clustering) imports...")
sys.path.insert(0, '/workspace/ClustGDD')

from models.gcn import GCN, MLP
from models.sgc_multi import SGC
from sklearn.cluster import KMeans, MiniBatchKMeans
print("  All ClustGDD model imports OK!")

nfeat, nhid, nclass = 100, 64, 5
gcn = GCN(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=0.5,
          weight_decay=5e-4, nlayers=2, device='cpu').to('cpu')
print(f"  GCN model created: {sum(p.numel() for p in gcn.parameters())} parameters")

mlp = MLP(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=0.5,
          weight_decay=5e-4, nlayers=2, lr=0.01, device='cpu').to('cpu')
print(f"  MLP model created: {sum(p.numel() for p in mlp.parameters())} parameters")

# Test forward pass
x = torch.randn(20, nfeat)
adj = torch.eye(20)
out = gcn(x, adj)
print(f"  GCN forward pass: input {x.shape} -> output {out.shape}")
assert out.shape == (20, nclass), "Output shape mismatch!"
print("  ClustGDD model verification PASSED!")

# ===== Part 4: Test PyG dataset loading =====
print("\n[4/4] Testing PyTorch Geometric dataset loading (Cora)...")
from torch_geometric.datasets import Planetoid
dataset_cora = Planetoid(root='/tmp/pyg_data', name='Cora')
data = dataset_cora[0]
print(f"  Cora dataset: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features")
print(f"  Classes: {dataset_cora.num_classes}")
print("  PyG dataset loading PASSED!")

print("\n" + "=" * 60)
print("ALL TESTS PASSED - Development environment is ready!")
print("=" * 60)
