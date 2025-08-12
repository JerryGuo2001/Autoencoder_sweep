# AE-style feedforward network trained only from direct pairings
# Objective: next-item prediction via softmax CE; no path-length supervision
import os, hashlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import networkx as nx, itertools, random, math
from scipy.spatial.distance import cdist
import pandas as pd
from typing import List
import sys

partition = int(sys.argv[1])
callback= int(sys.argv[2])

# ------------------ utils ------------------
def set_seed(s=123):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    try:
        if torch.backends.mps.is_available(): return torch.device("mps")
    except Exception: pass
    return torch.device("cpu")

def cosine_sim(a,b): return 1 - cdist(a,b,metric="cosine")
def shortest_paths(adj): return np.asarray(nx.floyd_warshall_numpy(nx.from_numpy_array(adj)), dtype=float)
def edges_from_adj(adj): return np.array(list(nx.from_numpy_array(adj).edges()),dtype=int)

# ------------------ data ------------------
def make_direct_pairs(edges, n_reps):
    """Repeat each edge n_reps times and shuffle (direct pairs only)."""
    idx = np.repeat(np.arange(len(edges)), n_reps)
    np.random.shuffle(idx)
    e = edges[idx]
    n = int(edges.max()+1)
    X = np.eye(n, dtype=np.float32)[e[:,0]]
    Y = e[:,1].astype(np.int64)  # class indices for CE
    return X, Y

class PairDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ------------------ blocked partition (cached) ------------------
def _edges_hash(adj: np.ndarray, nlists: int, seed: int) -> str:
    h = hashlib.md5()
    h.update(adj.astype(np.uint8).tobytes())
    h.update(str(nlists).encode())
    h.update(str(seed).encode())
    return h.hexdigest()[:16]

def find_blocks_once(adj: np.ndarray, nlists: int = 4, tries: int = 10000, seed: int = 123) -> List[np.ndarray]:
    """
    Partition edges into <= nlists blocks; each block is a matching (no node repeats).
    Returns a list of 1D numpy arrays of edge indices (variable lengths).
    Pure in-memory: no disk cache, no external files.
    """
    G = nx.from_numpy_array(adj)
    edges = np.array(list(G.edges()), dtype=int)
    nE = len(edges)
    rng = np.random.default_rng(seed)

    def try_one():
        # Greedy edge-coloring with exactly nlists colors (first-fit)
        used_nodes = [set() for _ in range(nlists)]
        blocks = [[] for _ in range(nlists)]

        order = np.arange(nE)
        rng.shuffle(order)
        for ei in order:
            u, v = edges[ei]
            placed = False
            for c in range(nlists):
                if (u not in used_nodes[c]) and (v not in used_nodes[c]):
                    blocks[c].append(ei)
                    used_nodes[c].update((u, v))
                    placed = True
                    break
            if not placed:
                return None
        return [np.array(b, dtype=int) for b in blocks]

    for _ in range(tries):
        out = try_one()
        if out is not None:
            return out

    raise RuntimeError("Could not find a block partition. Try increasing 'nlists' or 'tries'.")


# ------------------ model ------------------
class AEStyleCE(nn.Module):
    def __init__(self, n_items, L1=12, L2=36, n_hidden=12, p_drop=0.05,
                 init="he", scale_init=30.0, margin=0.20):
        super().__init__()
        self.E1 = nn.Linear(n_items, L1)
        self.E2 = nn.Linear(L1, L2)
        self.E3 = nn.Linear(L2, n_hidden)
        self.dropout = nn.Dropout(p_drop)
        self.W = nn.Parameter(torch.randn(n_hidden, n_items) * 0.02)
        self.log_scale = nn.Parameter(torch.log(torch.tensor(scale_init, dtype=torch.float32)))
        self.margin = margin
        self._init(init)

    def _init(self, how):
        def init_w(m):
            if isinstance(m, nn.Linear):
                if how=="xavier": nn.init.xavier_uniform_(m.weight)
                elif how=="he": nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif how=="lecun": nn.init.normal_(m.weight, 0, (1/m.weight.size(1))**0.5)
                elif how=="orthogonal": nn.init.orthogonal_(m.weight)
                else: raise ValueError(how)
                if m.bias is not None: nn.init.zeros_(m.bias)
        self.apply(init_w)

    def encode(self, x):
        x = F.relu(self.E1(x))
        x = self.dropout(F.relu(self.E2(x)))
        h = self.E3(x)
        return h

    def forward(self, x, y=None):
        h = self.encode(x)
        h_norm = F.normalize(h, dim=1)
        W_norm = F.normalize(self.W, dim=0)
        cos = torch.matmul(h_norm, W_norm)               # (B, C)
        s = self.log_scale.exp()
        if y is None:
            return cos * s, h
        logits = cos.clone()
        logits[torch.arange(logits.size(0), device=logits.device), y] -= self.margin
        return logits * s, h


# ------------------ training ------------------
def train_next_item(model, loader, n_epochs=120, lr=2.5e-4, wd=5e-3, patience=18, dev=None, label_smoothing=0.0):
    dev = dev or device()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing) if label_smoothing>0 else nn.CrossEntropyLoss()

    best, wait, best_state = math.inf, 0, None
    for _ in range(n_epochs):
        model.train(); losses = []
        for X, y in loader:
            X, y = X.to(dev), y.to(dev)
            logits, _ = model(X, y=y)          # <-- pass y
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
        m = float(np.mean(losses)); sched.step(m)
        if m < best - 1e-6:
            best, wait, best_state = m, 0, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                model.load_state_dict(best_state); break
    return best


# ------------------ evaluation ------------------
def hidden_activations(model, n_items, dev):
    I = torch.eye(n_items, dtype=torch.float32, device=dev)
    model.eval()
    with torch.no_grad():
        _, H = model(I)
    return H.detach().cpu().numpy()

def relative_distance_accuracy(n_items, sims, D):
    buckets = {1:[], 2:[], 3:[], 4:[]}
    for i1, i2, i3 in itertools.permutations(range(n_items), 3):
        # exclude direct or identical relations
        if D[i1,i2] <= 1 or D[i3,i2] <= 1 or D[i1,i3] <= 1:
            continue
        d12, d32 = D[i1,i2], D[i3,i2]
        dd = int(abs(d12 - d32))
        if dd not in buckets: continue
        correct = 0 if d12 < d32 else 1  # nearer in graph distance
        s12, s32 = sims[i1,i2], sims[i3,i2]
        model_choice = 0 if s12 > s32 else 1  # higher similarity -> nearer
        buckets[dd].append(1 if model_choice==correct else 0)
    return {k: (100*np.mean(v) if len(v) else np.nan) for k,v in buckets.items()}

# ------------------ experiment ------------------
def run(adj, L2_grid, n_models=10, regime="I", seed=123,
        L1=12, n_hidden=3, p_drop=0.05, lr=2.5e-4, wd=1.62e-2,
        n_epochs=120, batch_size=1):  # <-- batch_size defaults to 1
    set_seed(seed)
    dev = device()
    n_items = adj.shape[0]
    D = shortest_paths(adj)
    edges = edges_from_adj(adj)

    blocks = None
    if regime == "B":
        blocks = find_blocks_once(adj, nlists=4, tries=10000, seed=seed)

    records = []
    for L2 in L2_grid:
        for rep in range(n_models):
            set_seed(seed + 31*rep + 7*L2)

            if regime == "I":
                # Balanced interleaved exposure (direct pairs only)
                X, Y = make_direct_pairs(edges, n_reps=60)

            elif regime == "B":
                # Replay each block (matching) repeatedly before moving on
                Xs, Ys = [], []
                eye = np.eye(n_items, dtype=np.float32)
                reps_per_edge = 4  # 4 repeat per edge

                for block in blocks:
                    sub = edges[block]  # (k,2) node pairs
                    if len(sub) == 0:
                        continue
                    idx = np.repeat(np.arange(len(sub)), reps_per_edge)
                    np.random.shuffle(idx)
                    e = sub[idx]
                    Xs.append(eye[e[:,0]])
                    Ys.append(e[:,1].astype(np.int64))

                if len(Xs) == 0:
                    X, Y = make_direct_pairs(edges, n_reps=60)
                else:
                    X, Y = np.vstack(Xs), np.concatenate(Ys)

            else:
                raise ValueError("regime must be 'I' or 'B'")

            ds = PairDataset(X, Y)
            # Online-style: batch_size=1, keep order (shuffle=False), and keep all items (drop_last=False)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

            model = AEStyleCE(n_items, L1=L1, L2=L2, n_hidden=n_hidden, p_drop=p_drop, init="he").to(dev)
            best_loss = train_next_item(model, loader, n_epochs=n_epochs, lr=lr, wd=wd, patience=14, dev=dev, label_smoothing=0.0)

            H = hidden_activations(model, n_items, dev)
            # cosine similarity of normalized codes for judgment
            Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
            sims = cosine_sim(Hn, Hn)
            acc = relative_distance_accuracy(n_items, sims, D)

            records.append({
                "regime": regime, "L2": L2, "rep": rep,
                "acc_d1": acc[1], "acc_d2": acc[2], "acc_d3": acc[3], "acc_d4": acc[4],
                "best_loss": best_loss
            })
            print(f"[{regime}] L2={L2:>3} rep={rep} | d1={acc[1]:.1f} d2={acc[2]:.1f} d3={acc[3]:.1f} d4={acc[4]:.1f}  loss={best_loss:.4f}")
    return pd.DataFrame.from_records(records)

# ------------------ run on your graph ------------------
if __name__ == "__main__":
    set_seed(123)
    Gedges = np.array([
        [0,1,0,0,0,0,0,0,0,0,0,0],
        [1,0,1,1,0,0,0,0,0,0,0,0],
        [0,1,0,1,1,1,0,0,0,0,0,0],
        [0,1,1,0,0,0,1,1,0,0,0,0],
        [0,0,1,0,0,0,0,1,0,1,0,0],
        [0,0,1,0,0,0,0,0,1,1,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,1,1,0],
        [0,0,0,0,1,1,0,0,1,0,1,0],
        [0,0,0,0,0,0,0,0,1,1,0,1],
        [0,0,0,0,0,0,0,0,0,0,1,0],
    ], dtype=np.int32)

    L2_grid = [6, 12, 36, 72, 108, 144, 180, 216, 252, 288, 324]
    df = run(
        Gedges, L2_grid, n_models=50, regime="B",
        seed=125, L1=12, n_hidden=12, p_drop=0.02,
        lr=2.5e-4, wd=5e-3, n_epochs=100, batch_size=1
    )
    print(df.groupby(["regime","L2"])[["acc_d1","acc_d2","acc_d3","acc_d4","best_loss"]].mean().round(1))

    # save to output_data/<callback>_<partition>_<arrayid>.csv
    os.makedirs("output_data", exist_ok=True)
    out_path = os.path.join("output_data", f"Block_{callback}_{partition}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
