r"""
Graph Distillation for Recommendation (Bipartite User-Item) by adapting ClustGDD + Rankformer(BPR).

What is kept from ClustGDD:
- The *clustering/compression* idea: merge original nodes into super-nodes via K-Means, then
  build a condensed graph by aggregating edges (counts/weights).
- The *refinement/synthesis* idea: learn a small condensed graph (node representations + edge weights)
  by optimizing an objective on the condensed graph (here we replace CE classification loss with BPR).

What is borrowed from Rankformer:
- The *ranking objective* (BPR loss) with (user, positive item, negative item) triplet sampling.

This script is intentionally self-contained (no labels, no node classification pipeline).

Run example (PowerShell):
  python .\ClustGDD\distill_recsys.py --data_dir .\Rankformer\data --dataset Ali-Display --device cuda
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Some Windows/conda environments ship mismatched torch.onnx / torch._dynamo pieces.
# Optimizers may import torch._dynamo via torch._compile, which can fail if ONNX internals are broken.
# Disabling Dynamo keeps eager training working and is safe for this script.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Data loading (Rankformer-style)
# -----------------------------

@dataclass
class RecDataset:
    """Minimal recommender dataset holder for bipartite graphs."""

    num_users: int
    num_items: int
    train_u: np.ndarray  # shape [E_train]
    train_i: np.ndarray  # shape [E_train]
    valid_u: np.ndarray
    valid_i: np.ndarray
    test_u: np.ndarray
    test_i: np.ndarray

    @property
    def num_edges_train(self) -> int:
        return int(self.train_u.shape[0])


def _read_ui_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read 'user item' per line (space separated), matching Rankformer data files."""
    arr = np.loadtxt(path, dtype=np.int64)
    if arr.ndim == 1:
        # single line
        arr = arr.reshape(1, 2)
    if arr.shape[1] < 2:
        raise ValueError(f"Bad interaction file format: {path} (need 2 columns: user item)")
    u = arr[:, 0].astype(np.int64, copy=False)
    i = arr[:, 1].astype(np.int64, copy=False)
    return u, i


def load_rankformer_dataset(data_dir: str, dataset: str) -> RecDataset:
    root = os.path.join(data_dir, dataset)
    train_path = os.path.join(root, "train.txt")
    valid_path = os.path.join(root, "valid.txt")
    test_path = os.path.join(root, "test.txt")
    for p in (train_path, valid_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    train_u, train_i = _read_ui_txt(train_path)
    valid_u, valid_i = _read_ui_txt(valid_path)
    test_u, test_i = _read_ui_txt(test_path)

    num_users = int(max(train_u.max(initial=0), valid_u.max(initial=0), test_u.max(initial=0)) + 1)
    num_items = int(max(train_i.max(initial=0), valid_i.max(initial=0), test_i.max(initial=0)) + 1)

    print(f"[data] {dataset}: {num_users} users, {num_items} items")
    print(
        f"[data] edges train/valid/test: "
        f"{train_u.shape[0]}/{valid_u.shape[0]}/{test_u.shape[0]}"
    )

    return RecDataset(
        num_users=num_users,
        num_items=num_items,
        train_u=train_u,
        train_i=train_i,
        valid_u=valid_u,
        valid_i=valid_i,
        test_u=test_u,
        test_i=test_i,
    )


def build_interaction_matrix(
    num_users: int, num_items: int, u: np.ndarray, i: np.ndarray, values: Optional[np.ndarray] = None
) -> sp.csr_matrix:
    """Build sparse user-item interaction matrix R (users x items)."""
    if values is None:
        values = np.ones_like(u, dtype=np.float32)
    mat = sp.coo_matrix((values.astype(np.float32), (u, i)), shape=(num_users, num_items))
    return mat.tocsr()


# -----------------------------
# Clustering phase (ClustGDD spirit)
# -----------------------------

def compute_svd_embeddings(
    R: sp.csr_matrix,
    dim: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute low-dim user/item embeddings from the interaction matrix using Truncated SVD.
    This is the recommendation analogue of ClustGDD "pretrained clustering embedding".
    """
    # Use scipy.sparse.linalg.svds (works on large sparse matrices).
    # Note: svds returns singular values in ascending order.
    from scipy.sparse.linalg import svds

    k = min(dim, min(R.shape) - 1)
    if k <= 0:
        raise ValueError(f"Cannot compute SVD with shape={R.shape} and dim={dim}")

    # svds is stochastic-ish depending on solver, set seed for numpy for reproducibility
    rs = np.random.RandomState(seed)
    _ = rs.rand(1)  # touch to silence "unused"

    U, S, VT = svds(R.astype(np.float64), k=k)
    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = U[:, idx]
    VT = VT[idx, :]

    # Standard choice: scale by sqrt(S)
    sqrtS = np.sqrt(np.maximum(S, 1e-12))
    user_emb = (U * sqrtS.reshape(1, -1)).astype(np.float32)
    item_emb = (VT.T * sqrtS.reshape(1, -1)).astype(np.float32)
    return user_emb, item_emb


def kmeans_cluster(
    X: np.ndarray,
    n_clusters: int,
    seed: int,
    minibatch: bool = True,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels, centers)."""
    if n_clusters <= 0:
        raise ValueError("n_clusters must be > 0")
    if n_clusters >= X.shape[0]:
        # degenerate: each point its own cluster (cap)
        n_clusters = max(1, min(n_clusters, X.shape[0]))

    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    if minibatch and X.shape[0] > 20000:
        km = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=seed, batch_size=batch_size, n_init="auto"
        )
    else:
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = km.fit_predict(Xs).astype(np.int64)
    centers = km.cluster_centers_.astype(np.float32)
    return labels, centers


def build_condensed_bipartite(
    train_u: np.ndarray,
    train_i: np.ndarray,
    u2cu: np.ndarray,
    i2ci: np.ndarray,
    num_cu: int,
    num_ci: int,
) -> sp.csr_matrix:
    """
    Aggregate edges (u,i) -> (cu,ci) to form condensed bipartite adjacency (cu x ci).
    Values are edge counts (can be treated as weights).
    """
    cu = u2cu[train_u]
    ci = i2ci[train_i]
    vals = np.ones_like(cu, dtype=np.float32)
    C = sp.coo_matrix((vals, (cu, ci)), shape=(num_cu, num_ci))
    C.sum_duplicates()
    return C.tocsr()


# -----------------------------
# Refinement stage (replace CE with BPR)
# -----------------------------

def _csr_row_to_set_list(mat: sp.csr_matrix) -> List[np.ndarray]:
    """For each row, store the column indices (positive items)."""
    pos = []
    for r in range(mat.shape[0]):
        start, end = mat.indptr[r], mat.indptr[r + 1]
        pos.append(mat.indices[start:end].copy())
    return pos


def sample_bpr_triplets_from_condensed(
    pos_items_by_user: List[np.ndarray],
    num_items: int,
    batch_size: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample (u, pos_i, neg_i) from *condensed* bipartite graph.
    - u: super-user id
    - pos_i: a connected super-item id
    - neg_i: a non-connected super-item id (rejection sampling)
    """
    num_users = len(pos_items_by_user)
    u = rng.randint(0, num_users, size=(batch_size,), dtype=np.int64)
    pos_i = np.empty((batch_size,), dtype=np.int64)
    neg_i = np.empty((batch_size,), dtype=np.int64)

    for idx in range(batch_size):
        uu = int(u[idx])
        pos_list = pos_items_by_user[uu]
        # Need: 0 < pos_list.size < num_items (has positives and at least one valid negative)
        tries = 0
        while tries < 50 and (pos_list.size == 0 or pos_list.size >= num_items):
            uu = int(rng.randint(0, num_users))
            pos_list = pos_items_by_user[uu]
            tries += 1
        u[idx] = uu

        if pos_list.size == 0:
            # give up: use random pos/neg (won't crash, but not meaningful)
            pos_i[idx] = int(rng.randint(0, num_items))
            neg_i[idx] = int(rng.randint(0, num_items))
            while neg_i[idx] == pos_i[idx]:
                neg_i[idx] = int(rng.randint(0, num_items))
            continue

        if pos_list.size >= num_items:
            # all items positive for this super-user, no valid negative; use pos from list, neg random
            pos_i[idx] = int(pos_list[rng.randint(0, pos_list.size)])
            neg_i[idx] = int(rng.randint(0, num_items))
            while neg_i[idx] == pos_i[idx]:
                neg_i[idx] = int(rng.randint(0, num_items))
            continue

        pi = int(pos_list[rng.randint(0, pos_list.size)])
        pos_i[idx] = pi

        # rejection sampling for negatives (must not be in pos_list and must differ from pos)
        neg = int(rng.randint(0, num_items))
        tries = 0
        while tries < 50 and (np.isin(neg, pos_list) or neg == pi):
            neg = int(rng.randint(0, num_items))
            tries += 1
        neg_i[idx] = neg

    return u, pos_i, neg_i


class LightGCNCondensed(nn.Module):
    """
    A compact LightGCN-like encoder on the condensed bipartite graph.
    We make edge weights learnable (ClustGDD-style refinement on adjacency),
    while node representations are learnable embeddings (Rankformer-style).
    """

    def __init__(
        self,
        num_cu: int,
        num_ci: int,
        dim: int,
        num_layers: int,
        edge_index: torch.Tensor,  # [2, E] in COO over (cu, ci)
        edge_weight_init: torch.Tensor,  # [E]
        device: torch.device,
    ):
        super().__init__()
        self.num_cu = int(num_cu)
        self.num_ci = int(num_ci)
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.device = device

        self.user_emb = nn.Embedding(self.num_cu, self.dim)
        self.item_emb = nn.Embedding(self.num_ci, self.dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        # Learnable edge logits; use softplus to keep weights positive & stable.
        self.edge_index = edge_index.to(device)
        # Stable inverse-softplus for initialization:
        # softplus(x)=log(1+exp(x)); inverse is log(exp(y)-1) but overflows for large y.
        y = edge_weight_init.clamp_min(1e-6)
        inv_sp = torch.where(y > 20.0, y, torch.log(torch.expm1(y)))
        self.edge_logit = nn.Parameter(inv_sp)

        # Optional "CAAR-like" refinement: preference-aware offsets for super nodes
        self.user_delta = nn.Parameter(torch.zeros(self.num_cu, self.dim, device=device))
        self.item_delta = nn.Parameter(torch.zeros(self.num_ci, self.dim, device=device))

    def edge_weight(self) -> torch.Tensor:
        return F.softplus(self.edge_logit) + 1e-8

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN propagation on bipartite condensed graph with learnable edge weights.
        Returns final (user_z, item_z).
        """
        u0 = self.user_emb.weight + self.user_delta
        i0 = self.item_emb.weight + self.item_delta

        # Build normalized bi-adjacency for message passing:
        # For bipartite, we use D_u^{-1/2} * W * D_i^{-1/2}.
        cu = self.edge_index[0]  # [E]
        ci = self.edge_index[1]  # [E]
        w = self.edge_weight()   # [E]

        deg_u = torch.zeros(self.num_cu, device=self.device).index_add_(0, cu, w)
        deg_i = torch.zeros(self.num_ci, device=self.device).index_add_(0, ci, w)
        norm = w / (torch.sqrt(deg_u[cu] + 1e-8) * torch.sqrt(deg_i[ci] + 1e-8))

        u = u0
        it = i0
        u_layers = [u]
        i_layers = [it]
        for _ in range(self.num_layers):
            # u <- sum_{(u,i)} norm * i
            u_msg = torch.zeros_like(u).index_add_(0, cu, it[ci] * norm.unsqueeze(1))
            # i <- sum_{(u,i)} norm * u
            i_msg = torch.zeros_like(it).index_add_(0, ci, u[cu] * norm.unsqueeze(1))
            u, it = u_msg, i_msg
            u_layers.append(u)
            i_layers.append(it)

        # layer-mean (LightGCN)
        u_out = torch.stack(u_layers, dim=0).mean(dim=0)
        i_out = torch.stack(i_layers, dim=0).mean(dim=0)
        return u_out, i_out

    def bpr_loss(
        self,
        u: torch.Tensor,
        pos_i: torch.Tensor,
        neg_i: torch.Tensor,
        reg_lambda: float = 1e-4,
    ) -> torch.Tensor:
        """Rankformer-style BPR: softplus(score(u,neg) - score(u,pos))."""
        u_z, i_z = self.propagate()

        u_vec = u_z[u]
        pos_vec = i_z[pos_i]
        neg_vec = i_z[neg_i]

        pos_score = (u_vec * pos_vec).sum(dim=-1)
        neg_score = (u_vec * neg_vec).sum(dim=-1)
        loss_rank = F.softplus(neg_score - pos_score).mean()

        # lightweight regularization on base embeddings + deltas + edge weights
        reg = (
            self.user_emb(u).norm(2).pow(2)
            + self.item_emb(pos_i).norm(2).pow(2)
            + self.item_emb(neg_i).norm(2).pow(2)
        ) / max(1, u.shape[0])
        reg = reg + 1e-3 * (self.user_delta.norm(2).pow(2) + self.item_delta.norm(2).pow(2)) / (
            self.num_cu + self.num_ci
        )
        reg = reg + 1e-6 * self.edge_weight().norm(2).pow(2)

        return loss_rank + reg_lambda * reg


def condensed_csr_to_edge_index(C: sp.csr_matrix, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert condensed cu x ci CSR to COO tensors (edge_index, edge_weight_init)."""
    C = C.tocoo()
    cu = torch.from_numpy(C.row.astype(np.int64))
    ci = torch.from_numpy(C.col.astype(np.int64))
    w = torch.from_numpy(C.data.astype(np.float32))
    edge_index = torch.stack([cu, ci], dim=0).to(device)
    edge_weight = w.to(device)
    return edge_index, edge_weight


# -----------------------------
# Optimizer (manual Adam to avoid torch._dynamo / torch._compile issues on some envs)
# -----------------------------

@torch.no_grad()
def manual_adam_step(
    params: List[torch.nn.Parameter],
    state: Dict[int, Dict[str, torch.Tensor]],
    lr: float,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    step: int = 1,
) -> None:
    """
    Minimal Adam update to avoid importing torch.optim (which may pull torch._dynamo/onnx in broken installs).
    This is sufficient for the thesis prototype script.
    """
    b1, b2 = betas
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        if weight_decay != 0.0:
            g = g.add(p, alpha=weight_decay)

        sid = id(p)
        if sid not in state:
            state[sid] = {
                "m": torch.zeros_like(p, memory_format=torch.preserve_format),
                "v": torch.zeros_like(p, memory_format=torch.preserve_format),
            }
        m = state[sid]["m"]
        v = state[sid]["v"]

        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)

        # bias correction
        m_hat = m / (1 - b1**step)
        v_hat = v / (1 - b2**step)
        p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


# -----------------------------
# Evaluation (optional)
# -----------------------------

@torch.no_grad()
def recall_at_k(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    train_R: sp.csr_matrix,
    test_u: np.ndarray,
    test_i: np.ndarray,
    k: int,
    device: torch.device,
    max_users: int = 5000,
) -> float:
    """
    Simple Recall@K on test interactions.
    - Masks training positives (do not recommend seen items).
    - Evaluates on a subset of users for speed (max_users).
    """
    num_users = user_emb.shape[0]
    num_items = item_emb.shape[0]

    # build ground truth list from test edges
    gt: Dict[int, List[int]] = {}
    for u, i in zip(test_u.tolist(), test_i.tolist()):
        gt.setdefault(int(u), []).append(int(i))
    users = np.array(sorted(gt.keys()), dtype=np.int64)
    if users.size == 0:
        return 0.0
    if users.size > max_users:
        users = users[:max_users]

    U = user_emb[torch.from_numpy(users).to(device)]  # [B, d]
    scores = U @ item_emb.t()  # [B, I]

    # mask train interactions
    for row_idx, u in enumerate(users.tolist()):
        start, end = train_R.indptr[u], train_R.indptr[u + 1]
        seen = train_R.indices[start:end]
        if seen.size > 0:
            scores[row_idx, torch.from_numpy(seen).to(device)] = -1e9

    _, topk = torch.topk(scores, k=min(k, num_items), dim=1)
    topk = topk.cpu().numpy()

    hit = 0
    total = 0
    for row_idx, u in enumerate(users.tolist()):
        true_items = set(gt[u])
        if not true_items:
            continue
        rec_items = set(topk[row_idx].tolist())
        hit += len(true_items & rec_items)
        total += len(true_items)
    return float(hit / max(1, total))


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_dir", type=str, default=os.path.join("Rankformer", "data"))
    parser.add_argument("--dataset", type=str, default="Ali-Display")

    # distillation: clustering/compression
    parser.add_argument("--reduction_rate", type=float, default=0.1, help="keep rate for users/items (per-side).")
    parser.add_argument("--svd_dim", type=int, default=64)
    parser.add_argument("--kmeans_minibatch", action="store_true")
    parser.add_argument("--kmeans_batch_size", type=int, default=2048)

    # refinement: BPR training on condensed graph
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--lgn_layers", type=int, default=2)
    parser.add_argument("--refine_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--reg_lambda", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)

    # knowledge distillation (teacher → student on condensed graph)
    parser.add_argument(
        "--teacher_path",
        type=str,
        default=None,
        help="Path to teacher Rankformer embeddings (.pt with user_emb/item_emb).",
    )
    parser.add_argument(
        "--kd_lambda",
        type=float,
        default=0.01,
        help="Weight for KD loss (MSE between student and aggregated teacher embeddings).",
    )

    # runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_topk", type=int, default=20)
    parser.add_argument("--eval_max_users", type=int, default=5000)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[env] device={device}")

    # 1) Load recsys dataset (Rankformer format)
    ds = load_rankformer_dataset(args.data_dir, args.dataset)

    # 2) Build interaction matrix R for clustering
    R_train = build_interaction_matrix(ds.num_users, ds.num_items, ds.train_u, ds.train_i)

    # 3) Compute low-dim embeddings and KMeans clustering (ClustGDD clustering phase analogue)
    print("[cluster] computing SVD embeddings...")
    user_emb_np, item_emb_np = compute_svd_embeddings(R_train, dim=args.svd_dim, seed=args.seed)

    num_cu = max(1, int(math.ceil(ds.num_users * args.reduction_rate)))
    num_ci = max(1, int(math.ceil(ds.num_items * args.reduction_rate)))
    print(f"[cluster] target super nodes: users={num_cu}, items={num_ci}")

    print("[cluster] kmeans users...")
    u2cu, _ = kmeans_cluster(
        user_emb_np,
        n_clusters=num_cu,
        seed=args.seed,
        minibatch=args.kmeans_minibatch,
        batch_size=args.kmeans_batch_size,
    )
    print("[cluster] kmeans items...")
    i2ci, _ = kmeans_cluster(
        item_emb_np,
        n_clusters=num_ci,
        seed=args.seed,
        minibatch=args.kmeans_minibatch,
        batch_size=args.kmeans_batch_size,
    )

    # 4) Condense graph by aggregating edges (ClustGDD graph_compress analogue, but bipartite)
    print("[compress] building condensed bipartite graph...")
    C = build_condensed_bipartite(ds.train_u, ds.train_i, u2cu, i2ci, num_cu, num_ci)
    nnz = int(C.nnz)
    density = float(nnz / max(1, num_cu * num_ci))
    print(f"[compress] condensed edges: nnz={nnz}, density={density:.6f}")

    # 5) Optional: load teacher embeddings and aggregate to super-nodes (for KD)
    teacher_super_user_emb: Optional[torch.Tensor] = None
    teacher_super_item_emb: Optional[torch.Tensor] = None

    if args.teacher_path is not None:
        if not os.path.exists(args.teacher_path):
            raise FileNotFoundError(f"Teacher file not found: {args.teacher_path}")
        print(f"[teacher] loading teacher embeddings from: {args.teacher_path}")
        teacher_data = torch.load(args.teacher_path, map_location="cpu")
        if "user_emb" not in teacher_data or "item_emb" not in teacher_data:
            raise KeyError("teacher file must contain 'user_emb' and 'item_emb' tensors")

        teacher_user = teacher_data["user_emb"].to(device)  # [U, D_t]
        teacher_item = teacher_data["item_emb"].to(device)  # [I, D_t]

        if teacher_user.shape[0] != ds.num_users or teacher_item.shape[0] != ds.num_items:
            raise ValueError(
                f"Teacher emb size mismatch: "
                f"user_emb {teacher_user.shape[0]} vs {ds.num_users}, "
                f"item_emb {teacher_item.shape[0]} vs {ds.num_items}"
            )

        teacher_dim = int(teacher_user.shape[1])
        if teacher_dim != args.embed_dim:
            raise ValueError(
                f"Teacher embedding dim ({teacher_dim}) != student embed_dim ({args.embed_dim}). "
                f"Please re-train/save teacher with dim={args.embed_dim} or add a projection layer."
            )

        # Aggregate teacher embeddings to super-nodes via mean over cluster assignments
        print("[teacher] aggregating teacher embeddings to super-nodes (mean pooling)...")
        u2cu_t = torch.from_numpy(u2cu).long().to(device)  # [U]
        i2ci_t = torch.from_numpy(i2ci).long().to(device)  # [I]

        # Super-user means
        user_sum = torch.zeros(num_cu, teacher_dim, device=device)
        user_sum.index_add_(0, u2cu_t, teacher_user)
        user_cnt = torch.bincount(u2cu_t, minlength=num_cu).clamp_min(1).unsqueeze(1)
        teacher_super_user_emb = user_sum / user_cnt

        # Super-item means
        item_sum = torch.zeros(num_ci, teacher_dim, device=device)
        item_sum.index_add_(0, i2ci_t, teacher_item)
        item_cnt = torch.bincount(i2ci_t, minlength=num_ci).clamp_min(1).unsqueeze(1)
        teacher_super_item_emb = item_sum / item_cnt

        print("[teacher] KD will be applied during refinement.")

    # 6) Refinement: optimize condensed graph with BPR (+ optional KD) objective
    # Prepare positive lists for triplet sampling on condensed graph
    pos_items_by_user = _csr_row_to_set_list(C)
    edge_index, edge_weight_init = condensed_csr_to_edge_index(C, device=device)

    model = LightGCNCondensed(
        num_cu=num_cu,
        num_ci=num_ci,
        dim=args.embed_dim,
        num_layers=args.lgn_layers,
        edge_index=edge_index,
        edge_weight_init=edge_weight_init,
        device=device,
    ).to(device)

    # NOTE: do NOT use torch.optim here: some Windows/conda environments have a broken torch.onnx
    # which gets imported by torch._dynamo via torch._compile when constructing optimizers.
    params = [p for p in model.parameters() if p.requires_grad]
    adam_state: Dict[int, Dict[str, torch.Tensor]] = {}
    rng = np.random.RandomState(args.seed)

    # Build mapping from original node -> condensed embedding via cluster id
    u2cu_t = torch.from_numpy(u2cu).long().to(device)
    i2ci_t = torch.from_numpy(i2ci).long().to(device)

    # Evaluate baseline before refinement (using current random condensed embeddings)
    with torch.no_grad():
        cu_z, ci_z = model.propagate()
        user_z0 = cu_z[u2cu_t]  # [U, d]
        item_z0 = ci_z[i2ci_t]  # [I, d]
        r0 = recall_at_k(
            user_z0,
            item_z0,
            train_R=R_train,
            test_u=ds.test_u,
            test_i=ds.test_i,
            k=args.eval_topk,
            device=device,
            max_users=args.eval_max_users,
        )
    print(f"[eval] Recall@{args.eval_topk} before refinement: {r0:.6f}")

    print("[refine] optimizing condensed graph with BPR loss...")
    model.train()
    for ep in range(1, args.refine_epochs + 1):
        u_np, pos_np, neg_np = sample_bpr_triplets_from_condensed(
            pos_items_by_user=pos_items_by_user,
            num_items=num_ci,
            batch_size=args.batch_size,
            rng=rng,
        )
        u = torch.from_numpy(u_np).long().to(device)
        pos_i = torch.from_numpy(pos_np).long().to(device)
        neg_i = torch.from_numpy(neg_np).long().to(device)

        # Pure BPR loss on condensed graph
        loss_bpr = model.bpr_loss(u, pos_i, neg_i, reg_lambda=args.reg_lambda)

        # Optional KD loss: align student super-node embeddings with aggregated teacher embeddings
        if teacher_super_user_emb is not None and teacher_super_item_emb is not None:
            with torch.no_grad():
                # detach teacher so it is never updated
                t_user = teacher_super_user_emb.detach()
                t_item = teacher_super_item_emb.detach()
            stu_user_z, stu_item_z = model.propagate()
            loss_kd_user = F.mse_loss(stu_user_z, t_user)
            loss_kd_item = F.mse_loss(stu_item_z, t_item)
            loss_kd = loss_kd_user + loss_kd_item
            loss = loss_bpr + args.kd_lambda * loss_kd
        else:
            loss = loss_bpr

        loss.backward()
        manual_adam_step(params, adam_state, lr=args.lr, step=ep)
        model.zero_grad(set_to_none=True)

        if ep % args.log_every == 0 or ep == 1 or ep == args.refine_epochs:
            model.eval()
            with torch.no_grad():
                cu_z, ci_z = model.propagate()
                user_z = cu_z[u2cu_t]
                item_z = ci_z[i2ci_t]
                r = recall_at_k(
                    user_z,
                    item_z,
                    train_R=R_train,
                    test_u=ds.test_u,
                    test_i=ds.test_i,
                    k=args.eval_topk,
                    device=device,
                    max_users=args.eval_max_users,
                )
            model.train()
            print(f"[refine] ep={ep:04d} loss={loss.item():.6f} Recall@{args.eval_topk}={r:.6f}")

    # 6) Export distilled artifacts (condensed graph + mappings)
    out_dir = os.path.join("ClustGDD", "distilled_recsys", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # Save condensed graph (edge list with learned weights)
    with torch.no_grad():
        w = model.edge_weight().detach().cpu().numpy()
        ei = model.edge_index.detach().cpu().numpy()
    np.savez_compressed(
        os.path.join(out_dir, "condensed_graph.npz"),
        cu=ei[0],
        ci=ei[1],
        w=w,
        num_cu=np.int64(num_cu),
        num_ci=np.int64(num_ci),
    )
    np.save(os.path.join(out_dir, "u2cu.npy"), u2cu.astype(np.int64))
    np.save(os.path.join(out_dir, "i2ci.npy"), i2ci.astype(np.int64))

    # Save learned embeddings for super-nodes
    torch.save(
        {
            "user_emb": model.user_emb.weight.detach().cpu(),
            "item_emb": model.item_emb.weight.detach().cpu(),
            "user_delta": model.user_delta.detach().cpu(),
            "item_delta": model.item_delta.detach().cpu(),
        },
        os.path.join(out_dir, "condensed_embeddings.pt"),
    )
    print(f"[save] distilled artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()

