"""
CPU-compatible runner for Rankformer training.
Patches CUDA calls to run on CPU-only environments.
Uses a synthetic dataset for demonstration.
"""
import torch
import sys
import os
import numpy as np

# === CPU Patches ===
torch.cuda.is_available = lambda: False
torch.cuda.set_device = lambda x: None
torch.cuda.manual_seed_all = lambda x: None

# Create synthetic dataset
os.makedirs('/workspace/Rankformer/data/TestData', exist_ok=True)
np.random.seed(42)
n_users, n_items = 50, 30
for name, n in [('train.txt', 200), ('valid.txt', 50), ('test.txt', 50)]:
    users = np.random.randint(0, n_users, n)
    items = np.random.randint(0, n_items, n)
    with open(f'/workspace/Rankformer/data/TestData/{name}', 'w') as f:
        for u, i in zip(users, items):
            f.write(f'{u} {i}\n')

os.chdir('/workspace/Rankformer')
sys.path.insert(0, '/workspace/Rankformer/code')

sys.argv = [
    'run_rankformer_cpu.py',
    '--data=TestData',
    '--use_gcn',
    '--use_rankformer',
    '--rankformer_layers=2',
    '--rankformer_tau=0.5',
    '--max_epochs=50',
    '--valid_interval=10',
    '--device=0',
    '--seed=42',
]

from parse import args
args.device = torch.device('cpu')
args.data_dir = '/workspace/Rankformer/data/'
args.train_file = os.path.join(args.data_dir, args.data, 'train.txt')
args.valid_file = os.path.join(args.data_dir, args.data, 'valid.txt')
args.test_file = os.path.join(args.data_dir, args.data, 'test.txt')

print("=" * 60)
print("Rankformer Training (CPU)")
print("=" * 60)

from dataloader import MyDataset
dataset = MyDataset(args.train_file, args.valid_file, args.test_file, args.device)
print(f"\nDataset: {args.data}")
print(f"{dataset.num_users} users, {dataset.num_items} items")

from model import Model
model = Model(dataset).to(args.device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}\n")


def print_test_result():
    global best_epoch, test_pre, test_recall, test_ndcg
    print(f'===== Test Result (at epoch {best_epoch:d}) =====')
    for i, k in enumerate(args.topks):
        print(f'  ndcg@{k:d} = {test_ndcg[i]:f}, recall@{k:d} = {test_recall[i]:f}, pre@{k:d} = {test_pre[i]:f}')


best_valid_ndcg, best_epoch = 0., 0
test_pre = torch.zeros(len(args.topks))
test_recall = torch.zeros(len(args.topks))
test_ndcg = torch.zeros(len(args.topks))

# Initial validation
valid_pre, valid_recall, valid_ndcg = model.valid_func()
for i, k in enumerate(args.topks):
    print(f'[0/{args.max_epochs:d}] Valid: ndcg@{k:d} = {valid_ndcg[i]:f}, recall@{k:d} = {valid_recall[i]:f}')
if valid_ndcg[-1] > best_valid_ndcg:
    best_valid_ndcg, best_epoch = valid_ndcg[-1], 0
    test_pre, test_recall, test_ndcg = model.test_func()

# Training loop
for epoch in range(1, args.max_epochs + 1):
    train_loss = model.train_func()
    if epoch % args.show_loss_interval == 0:
        print(f'epoch {epoch:d}, train_loss = {train_loss.item():f}')

    if epoch % args.valid_interval == 0:
        valid_pre, valid_recall, valid_ndcg = model.valid_func()
        for i, k in enumerate(args.topks):
            print(f'[{epoch:d}/{args.max_epochs:d}] Valid: ndcg@{k:d} = {valid_ndcg[i]:f}, recall@{k:d} = {valid_recall[i]:f}')
        if valid_ndcg[-1] > best_valid_ndcg:
            best_valid_ndcg, best_epoch = valid_ndcg[-1], epoch
            test_pre, test_recall, test_ndcg = model.test_func()
            print_test_result()
        elif epoch - best_epoch >= args.stopping_step * args.valid_interval:
            print(f'Early stopping at epoch {epoch}')
            break

print('\n' + '=' * 60)
print('Training done.')
print_test_result()
print('=' * 60)
