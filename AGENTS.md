# AGENTS.md

## Project Overview

This is a research monorepo combining two graph-based recommendation system projects, each as a git submodule:

- **ClustGDD** (`ClustGDD/`) — Graph data distillation via clustering (KDD 2025). Source: [HKBU-LAGAS/ClustGDD](https://github.com/HKBU-LAGAS/ClustGDD)
- **Rankformer** (`Rankformer/`) — Graph transformer for recommendation (WWW 2025). Source: [StupidThree/Rankformer](https://github.com/StupidThree/Rankformer)

## Cursor Cloud specific instructions

### Submodule initialization

The `.gitmodules` file maps the two submodules to their upstream repos. After cloning, run:

```bash
git submodule init && git submodule update --recursive
```

### CPU-only environment

This cloud VM has **no GPU**. Both projects default to CUDA devices. When running training scripts:

- **Rankformer**: `parse.py` hardcodes `args.device = torch.device(f'cuda:{args.device:d}')`. Override it to `torch.device('cpu')` after import. The `dataloader.py` also moves tensors to the device at load time, so all data/model must be on CPU.
- **ClustGDD**: `train_clustgdd_transduct.py` calls `torch.cuda.set_device(args.gpu_id)` at startup and uses `device='cuda:{gpu_id}'`. The `test_with_val` method also uses `.cuda()` directly. These must be patched for CPU usage.

### Dependencies

All Python dependencies are installed system-wide via pip (no conda/venv). Key packages:

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.2.2+cpu | CPU-only, from PyTorch CPU index |
| torch-geometric | 2.7.0 | With scatter/sparse/cluster/spline-conv extensions |
| numpy | 1.26.4 | |
| pandas | 2.2.1 | Used by Rankformer |
| ogb | 1.3.6 | Used by ClustGDD for ogbn-arxiv |
| deeprobust | 0.2.11 | Used by ClustGDD (graph utils) |
| scikit-learn | latest | KMeans clustering for ClustGDD |
| scipy, matplotlib, networkx | latest | Various utilities |

### Running projects

- **Rankformer**: `cd Rankformer && python -u code/main.py --data=<dataset> --use_gcn --use_rankformer ...` — see `Rankformer/README.md` for full examples.
- **ClustGDD**: `cd ClustGDD && sh main_transduct.sh` or `sh main_induct.sh` — see `ClustGDD/README.md` for details.

### Lint

No project-specific linter config exists. Use `flake8 --max-line-length=200` for basic checks. No automated tests exist (typical for academic ML research repos).

### Known gotchas

1. `deeprobust` via pip pulls in a full PyTorch with CUDA. The update script reinstalls `torch` CPU-only after deeprobust to fix this.
2. ClustGDD's `utils.py` has a hardcoded path `path = osp.join('xxx/data', name)` for dataset storage. The Planetoid datasets (cora, citeseer) are downloaded by PyG automatically, so this path is only used as a cache directory.
3. PyTorch Geometric extension wheels (scatter, sparse, cluster, spline-conv) must match the PyTorch version. They are installed from `https://data.pyg.org/whl/torch-2.2.0+cpu.html`.
