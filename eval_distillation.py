"""
Comprehensive evaluation of ClustGDD graph distillation effectiveness.
Tests multiple datasets and reduction rates, compares with full-data GCN baseline.
"""
import torch
import sys
import os
import json
import time

# === CPU Patches ===
torch.cuda.is_available = lambda: False
torch.cuda.set_device = lambda x: None
torch.cuda.manual_seed = lambda x: None
torch.cuda.manual_seed_all = lambda x: None
torch.cuda.max_memory_allocated = lambda *a, **k: 0
_orig_cuda = torch.Tensor.cuda
torch.Tensor.cuda = lambda self, *a, **k: self

os.chdir('/workspace/ClustGDD')
sys.path.insert(0, '/workspace/ClustGDD')

import numpy as np
import random
import argparse
import torch.nn.functional as F
from copy import deepcopy

from utils import *
from utils_clustgdd import *
from clustgdd_agent_transduct import ClustGDD
import deep_robust_utils as dr_utils
from models.gcn import GCN

device = 'cpu'

# ============================================================
# Part 1: GCN Baseline (train on FULL original graph)
# ============================================================
def run_gcn_baseline(dataset_name, runs=5):
    """Train GCN directly on the full original graph (no distillation)."""
    random.seed(15)
    np.random.seed(15)
    torch.manual_seed(15)

    data_full = get_dataset(dataset_name, normalize_features=True)
    data = Transd2Ind(data_full, keep_ratio=1.0)

    results = []
    for r in range(runs):
        model = GCN(nfeat=data.feat_full.shape[1], nhid=256, nclass=data.nclass,
                     dropout=0.5, weight_decay=5e-4, nlayers=2, device=device).to(device)

        features, adj, labels = dr_utils.to_tensor(
            data.feat_full, data.adj_full, data.labels_full, device=device)
        adj_norm = dr_utils.normalize_adj_tensor(adj, sparse=True)
        labels_train = torch.LongTensor(data.labels_train).to(device)
        labels_test = torch.LongTensor(data.labels_test).to(device)
        labels_val = torch.LongTensor(data.labels_val).to(device)
        idx_train = data.idx_train
        idx_val = data.idx_val
        idx_test = data.idx_test

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        best_val_acc = 0
        best_weights = None

        for epoch in range(600):
            model.train()
            optimizer.zero_grad()
            output = model(features, adj_norm)
            loss = F.nll_loss(output[idx_train], labels[idx_train])
            loss.backward()
            optimizer.step()

            if epoch == 300:
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

            with torch.no_grad():
                model.eval()
                output = model(features, adj_norm)
                val_acc = dr_utils.accuracy(output[idx_val], labels[idx_val])
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = deepcopy(model.state_dict())

        model.load_state_dict(best_weights)
        model.eval()
        output = model(features, adj_norm)
        test_acc = dr_utils.accuracy(output[idx_test], labels[idx_test]).item()
        results.append(test_acc)

    return np.mean(results), np.std(results)


# ============================================================
# Part 2: ClustGDD Distillation
# ============================================================
def run_clustgdd(dataset_name, reduction_rate, config):
    """Run ClustGDD distillation and evaluate."""
    random.seed(15)
    np.random.seed(15)
    torch.manual_seed(15)

    data_full = get_dataset(dataset_name, normalize_features=True)
    data = Transd2Ind(data_full, keep_ratio=1.0)

    args = argparse.Namespace(
        gpu_id=0, dataset=dataset_name, nlayers=2, hidden=256,
        weight_decay=0.0, dropout=0.0, normalize_features=True,
        keep_ratio=1.0, reduction_rate=reduction_rate, seed=15, sgc=1, save=0,
        gctype='clustgdd', prop_num=config['prop_num'], alpha=config['alpha'],
        prehidden=256, predropout=config['predropout'], prewd=5e-4, prelr=0.01,
        preep=config['preep'], prenlayers=2, cluster_minibatch=1000,
        sp_ratio=config['sp_ratio'], sp_type='attaw',
        postep=config['postep'], postprop_num=config['postprop_num'],
        postlr_feat=1e-4, postlr_adj=1e-4, postlr_model=1e-2,
        postwd_feat=5e-4, postwd_adj=5e-4, postwd_model=5e-4,
        frcoe=config['frcoe'], predcoe=config['predcoe'], csttemp=0.5,
        w1=config.get('w1', 0.1), w2=1., no_refinement=False, no_adjsyn=False,
        save_pretrained_output=False, save_syn_output=False,
        save_norf=False, notopo=False, tm_rec=True,
    )

    agent = ClustGDD(data, args, device=device)

    t0 = time.time()
    # --- Inline the training to capture intermediate results ---
    feat_syn, labels_syn, cluster_labels, adj_norm, features, adj, labels, \
        target_feat, idx_train, idx_val, ebd = agent.pretrained_clustering(data)
    t_pretrain = time.time() - t0

    sparsed_graph_list = agent.graph_sparse(adj_norm, ratio=args.sp_ratio, ebd=ebd, sp_type=args.sp_type)
    compressed_graph_list, adj_syn = agent.graph_compress(cluster_labels, adj_norm, sparsed_graph_list)
    feat_syn = agent.graph_refusion(target_feat, idx_train, idx_val, labels, feat_syn, compressed_graph_list, labels_syn)
    t_distill = time.time() - t0

    adj_syn = adj_syn.detach().to_dense()
    adj_syn_norm = dr_utils.normalize_adj_tensor(adj_syn)

    agent.feat_syn = feat_syn
    agent.labels_syn = labels_syn
    agent.adj_syn = adj_syn_norm

    # Evaluate: train GCN on synthetic graph, test on original
    n_syn = feat_syn.shape[0]
    n_orig = data.feat_full.shape[0]
    compression_ratio = n_syn / len(data.idx_train)

    res = []
    for i in range(5):
        r = agent.test_with_val(i, verbose=False)
        res.append(r)
    res = np.array(res)
    train_acc_mean, test_acc_mean = res.mean(0)
    train_acc_std, test_acc_std = res.std(0)

    return {
        'dataset': dataset_name,
        'reduction_rate': reduction_rate,
        'syn_nodes': n_syn,
        'orig_train_nodes': len(data.idx_train),
        'compression': f"{compression_ratio:.1%}",
        'train_acc': f"{train_acc_mean:.4f} ± {train_acc_std:.4f}",
        'test_acc': f"{test_acc_mean:.4f} ± {test_acc_std:.4f}",
        'test_acc_val': test_acc_mean,
        'test_acc_std_val': test_acc_std,
        'pretrain_time': t_pretrain,
        'distill_time': t_distill,
    }


# ============================================================
# Configuration per dataset/rate (from main_transduct.sh)
# ============================================================
CONFIGS = {
    'cora': {
        1.0: dict(prop_num=20, alpha=0.8, predropout=0.6, sp_ratio=0.4, preep=80, postep=2000, postprop_num=2, frcoe=0.01, predcoe=1.0, w1=0.1),
        0.5: dict(prop_num=5, alpha=0.8, predropout=0.6, sp_ratio=0.4, preep=80, postep=2000, postprop_num=2, frcoe=0.01, predcoe=1.0, w1=0.1),
        0.25: dict(prop_num=5, alpha=0.8, predropout=0.7, sp_ratio=0.06, preep=80, postep=2000, postprop_num=2, frcoe=0.01, predcoe=1.0, w1=0.1),
    },
    'citeseer': {
        1.0: dict(prop_num=2, alpha=0.5, predropout=0.7, sp_ratio=0.2, preep=200, postep=200, postprop_num=1, frcoe=0.01, predcoe=0.9, w1=0.1),
        0.5: dict(prop_num=2, alpha=0.5, predropout=0.8, sp_ratio=0.21, preep=100, postep=200, postprop_num=1, frcoe=0.01, predcoe=0.05, w1=0.1),
        0.25: dict(prop_num=2, alpha=0.8, predropout=0.7, sp_ratio=0.06, preep=120, postep=80, postprop_num=1, frcoe=0.01, predcoe=1.0, w1=0.1),
    },
}

# For CPU speed, reduce postep
for ds in CONFIGS:
    for rate in CONFIGS[ds]:
        CONFIGS[ds][rate]['postep'] = min(CONFIGS[ds][rate]['postep'], 200)


# ============================================================
# Main evaluation
# ============================================================
if __name__ == '__main__':
    all_results = {}

    for dataset_name in ['cora', 'citeseer']:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset_name.upper()}")
        print(f"{'='*70}")

        # Baseline
        print(f"\n--- GCN Baseline (full data, no distillation) ---")
        baseline_mean, baseline_std = run_gcn_baseline(dataset_name, runs=5)
        print(f"  Baseline Test Accuracy: {baseline_mean:.4f} ± {baseline_std:.4f}")
        all_results[f'{dataset_name}_baseline'] = {
            'test_acc': f"{baseline_mean:.4f} ± {baseline_std:.4f}",
            'test_acc_val': baseline_mean,
        }

        # Distillation at various rates
        for rate in [1.0, 0.5, 0.25]:
            print(f"\n--- ClustGDD (reduction_rate={rate}) ---")
            config = CONFIGS[dataset_name][rate]
            result = run_clustgdd(dataset_name, rate, config)
            print(f"  Synthetic nodes: {result['syn_nodes']} / {result['orig_train_nodes']} train nodes ({result['compression']})")
            print(f"  Train Accuracy:  {result['train_acc']}")
            print(f"  Test Accuracy:   {result['test_acc']}")
            print(f"  Distill time:    {result['distill_time']:.2f}s")
            all_results[f'{dataset_name}_r{rate}'] = result

    # ============================================================
    # Summary table
    # ============================================================
    print(f"\n\n{'='*70}")
    print("  DISTILLATION EFFECTIVENESS SUMMARY")
    print(f"{'='*70}")

    for dataset_name in ['cora', 'citeseer']:
        baseline = all_results[f'{dataset_name}_baseline']
        print(f"\n  {dataset_name.upper()} (Baseline GCN: {baseline['test_acc']})")
        print(f"  {'Rate':<8} {'Syn Nodes':<12} {'Compression':<14} {'Test Acc':<22} {'vs Baseline':<14} {'Time':<10}")
        print(f"  {'-'*80}")
        for rate in [1.0, 0.5, 0.25]:
            r = all_results[f'{dataset_name}_r{rate}']
            diff = r['test_acc_val'] - baseline['test_acc_val']
            sign = '+' if diff >= 0 else ''
            print(f"  {rate:<8} {r['syn_nodes']:<12} {r['compression']:<14} {r['test_acc']:<22} {sign}{diff:.4f}        {r['distill_time']:.1f}s")

    print(f"\n{'='*70}")
