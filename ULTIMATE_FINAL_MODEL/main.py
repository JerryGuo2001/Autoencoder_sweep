#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# AE-style feedforward network trained only from direct pairings
# Objective: next-item prediction via softmax CE; no path-length supervision

import os, sys, hashlib, math, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import networkx as nx
from scipy.spatial.distance import cdist

# ===================== CLI =====================
# Usage: python script.py <partition> <callback> <arrayid>
# - partition selects init scheme and is also used in the seed
# - callback is just carried to the output filename
# - arrayid participates in the seed and filename
if len(sys.argv) < 4:
    print("Usage: python script.py <partition:int 1-4> <callback:int> <arrayid:int>")
    sys.exit(1)

partition = int(sys.argv[1])   # also used for init scheme & seed
callback  = int(sys.argv[2])   # used in output filename
arrayid   = int(sys.argv[3])   # used in seed & filename

# ===================== utils =====================
def set_seed(s=123):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def device():
    if torch.cuda.is_available(): 
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

def cosine_sim(a,b): 
    return 1 - cdist(a,b,metric="cosine")

def shortest_paths(adj): 
    return np.asarray(nx.floyd_warshall_numpy(nx.from_numpy_array(adj)), dtype=float)

def edges_from_adj(adj): 
    return np.array(list(nx.from_numpy_array(adj).edges()),dtype=int)

# ===================== data =====================
def make_direct_pairs(edges, n_reps):
    """Repeat each edge n_reps times (uniform over edges) and shuffle."""
    idx = np.repeat(np.arange(len(edges)), n_reps)
    np.random.shuffle(idx)
    e = edges[idx]
    n = int(edges.max()+1)
    X = np.eye(n, dtype=np.float32)[e[:,0]]
    Y = e[:,1].astype(np.int64)
    return X, Y

class PairDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ===================== blocked partition (cached) =====================
def _edges_hash(adj: np.ndarray, nlists: int, seed: int) -> str:
    h = hashlib.md5()
    h.update(adj.astype(np.uint8).tobytes())
    h.update(str(nlists).encode())
    h.update(str(seed).encode())
    return h.hexdigest()[:16]

def find_blocks_once(adj: np.ndarray, nlists: int = 4, tries: int = 10000, seed: int = 123,
                     cache_dir: str = "./blocked_cache"):
    """
    Partition edges into <= nlists blocks; each block is a matching (no node repeats).
    Returns a list of 1D numpy arrays of edge indices.
    Caches to disk so later runs reuse the same partition.
    """
    os.makedirs(cache_dir, exist_ok=True)
    G = nx.from_numpy_array(adj)
    edges = np.array(list(G.edges()), dtype=int)
    nE = len(edges)

    key = _edges_hash(adj, nlists, seed)
    cache_path = os.path.join(cache_dir, f"blocks_{key}.npy")

    # Reuse if exists
    if os.path.exists(cache_path):
        blocks_idx = np.load(cache_path, allow_pickle=True)
        return [np.array(b, dtype=int) for b in blocks_idx]

    rng = np.random.default_rng(seed)

    def try_one():
        used_nodes = [set() for _ in range(nlists)]
        blocks = [[] for _ in range(nlists)]
        order = np.arange(nE); rng.shuffle(order)
        for ei in order:
            u, v = edges[ei]; placed = False
            for c in range(nlists):
                if (u not in used_nodes[c]) and (v not in used_nodes[c]):
                    blocks[c].append(ei)
                    used_nodes[c].update([u, v]); placed = True
                    break
            if not placed:
                return None
        return [np.array(b, dtype=int) for b in blocks]

    for _ in range(tries):
        out = try_one()
        if out is not None:
            np.save(cache_path, np.array(out, dtype=object), allow_pickle=True)
            return out

    raise RuntimeError("Could not find a block partition. Try increasing 'nlists' or 'tries'.")

def make_block_dataloaders(edges, blocks, n_items, reps_per_edge=30,
                           batch_size=1, shuffle_within_block=False):
    """Return a list of DataLoaders, one per block, preserving block order."""
    eye = np.eye(n_items, dtype=np.float32)
    loaders = []
    for block in blocks:
        sub = edges[block]
        if len(sub) == 0: 
            continue
        idx = np.repeat(np.arange(len(sub)), reps_per_edge)
        if shuffle_within_block:
            np.random.shuffle(idx)
        e = sub[idx]
        Xb = eye[e[:, 0]]
        yb = e[:, 1].astype(np.int64)
        ds = PairDataset(Xb, yb)
        loaders.append(DataLoader(ds, batch_size=batch_size,
                                  shuffle=shuffle_within_block, drop_last=False))
    return loaders

# ===================== model =====================
class AEStyleCE(nn.Module):
    """
    Encoder: one-hot -> L1 -> L2 -> latent (n_hidden)
    Classifier head: linear from latent to items (no extra info).
    """
    def __init__(self, n_items, L1=12, L2=36, n_hidden=8, p_drop=0.15, init="he"):
        super().__init__()
        self.E1 = nn.Linear(n_items, L1)
        self.E2 = nn.Linear(L1, L2)
        self.E3 = nn.Linear(L2, n_hidden)
        self.dropout = nn.Dropout(p_drop)
        self.out = nn.Linear(n_hidden, n_items, bias=False)
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
    def forward(self, x):
        h = self.encode(x)
        logits = self.out(h)
        return logits, h

# ===================== training =====================
def train_next_item(model,
                    loader,                      # DataLoader or list of DataLoaders
                    n_epochs=50,
                    lr=1.5e-4,
                    wd=4.0e-2,
                    patience=8,
                    dev=None,
                    ls_eps: float = 0.10,
                    grad_clip: float = 1.0):
    """
    If `loader` is a single DataLoader => standard interleaved training.
    If `loader` is a sequence of DataLoaders => blocked training; iterates loaders in order each epoch.
    With batch_size=1 and drop_last=False, optimizer steps happen per (input, target) pair.
    """
    dev = dev or device()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, wait, best_state = math.inf, 0, None
    ce = nn.CrossEntropyLoss(label_smoothing=ls_eps)
    blocked_mode = isinstance(loader, (list, tuple))

    for ep in range(n_epochs):
        model.train()
        losses = []

        if blocked_mode:
            for ld in loader:
                for X, y in ld:
                    X, y = X.to(dev), y.to(dev)
                    logits, _ = model(X)
                    loss = ce(logits, y)
                    opt.zero_grad(); loss.backward()
                    if grad_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    losses.append(loss.item())
        else:
            for X, y in loader:
                X, y = X.to(dev), y.to(dev)
                logits, _ = model(X)
                loss = ce(logits, y)
                opt.zero_grad(); loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                losses.append(loss.item())

        m = float(np.mean(losses)) if losses else float("inf")
        if m < best - 1e-6:
            best, wait = m, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                if best_state is not None:
                    model.load_state_dict(best_state)
                break
    return best

# ===================== evaluation helpers =====================
def hidden_activations(model, n_items, dev):
    I = torch.eye(n_items, dtype=torch.float32, device=dev)
    model.eval()
    with torch.no_grad():
        _, H = model(I)
    return H.detach().cpu().numpy()

def relative_distance_accuracy(n_items, sims, D):
    # Buckets by |distance difference| (1..4), skipping any triple with a direct neighbor
    buckets = {1:[], 2:[], 3:[], 4:[]}
    import itertools
    for i1, i2, i3 in itertools.permutations(range(n_items), 3):
        if D[i1,i2] <= 1 or D[i3,i2] <= 1 or D[i1,i3] <= 1:
            continue
        d12, d32 = D[i1,i2], D[i3,i2]
        dd = int(abs(d12 - d32))
        if dd not in buckets: 
            continue
        # ground truth: nearer in graph distance to i2
        correct = 0 if d12 < d32 else 1
        s12, s32 = sims[i1,i2], sims[i3,i2]
        model_choice = 0 if s12 > s32 else 1
        buckets[dd].append(1 if model_choice==correct else 0)
    return {k: (100*np.mean(v) if len(v) else np.nan) for k,v in buckets.items()}

# ===================== graph (new task) =====================
nItems = 12
mapping = {0:'skate', 1:'chair', 2:'globe',3:'basket',4:'shades',5:'boat',6:'oven',7:'tree',
           8:'mailbox',9:'fan',10:'pawn',11:'couch'}
mappingN = {i: f"{name}_{i}" for i, name in mapping.items()}

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

# ===================== main experiment loop (HPC-style) =====================
def main():
    iw = 'he'

    # Hyperparameters (tuned for your “lower d1, higher d3” preference)
    lr = 1.5e-4
    wd = 4.0e-2
    p_drop = 0.15
    n_epochs = 50
    batch_size = 1
    n_reps_direct = 5       # TOTAL exposure per edge (no RW)
    ls_eps = 0.10
    grad_clip = 1.0
    n_models_per_combo = 100  # you said "train it for 100 model per"

    # Grid
    hidden_layer_widths = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108]
    tasks = ['I', 'B']  # Interleaved & Blocked (same as your old script)

    # Graph prep
    dev = device()
    set_seed(123)  # global deterministic baseline
    D = shortest_paths(Gedges)
    edges = edges_from_adj(Gedges)
    n_items = Gedges.shape[0]

    # Block partition once (for B)
    blocks = find_blocks_once(Gedges, nlists=4, tries=10000, seed=123, cache_dir="./blocked_cache")

    # Results accumulator (same fields as before)
    results = {'name':[], 'path':[], 'task':[], 'L2':[], '1':[], '2':[], '3':[], '4':[],
               'scores':[], 'end_loss':[], 'hidden':[], 'dists':[],
               'learn_rate':[], 'weight_decay':[], 'repetition':[],
               'initial_weight_type':[], 'initial_weights':[]}

    # Make sure output dir exists
    os.makedirs("output_data", exist_ok=True)
    data_dir = "./torchweights"
    os.makedirs(data_dir, exist_ok=True)

    # Per-combo training
    # seed_base is derived from BOTH partition and arrayid so different array tasks get different seeds
    seed_base = 10_000_000*partition + 10_000*arrayid + 97*callback

    for dataset_ID in tasks:
        for L2 in hidden_layer_widths:
            for model_i in range(n_models_per_combo):
                # Per-model seed
                seed_i = seed_base + model_i
                set_seed(seed_i)

                # Build dataset/loader(s)
                if dataset_ID == 'I':
                    # Interleaved: uniform repeats over edges, deterministic order
                    X, Y = make_direct_pairs(edges, n_reps=n_reps_direct)
                    ds = PairDataset(X, Y)
                    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
                elif dataset_ID == 'B':
                    # Blocked: present non-overlapping edge sets sequentially
                    loader = make_block_dataloaders(edges, blocks, n_items,
                                                    reps_per_edge=n_reps_direct,
                                                    batch_size=batch_size,
                                                    shuffle_within_block=False)
                else:
                    raise ValueError('dataset_ID must be "I" or "B"')

                # Model
                model_name  = f'{dataset_ID}_L2{L2}_m{model_i}'
                weight_path = os.path.join(data_dir, f'{model_name}.pt')
                model = AEStyleCE(n_items=n_items, L1=12, L2=L2, n_hidden=8, p_drop=p_drop, init=iw).to(dev)

                # Capture initial weights (same as your old script)
                initial_weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

                # Train
                best_loss = train_next_item(model, loader, n_epochs=n_epochs, lr=lr, wd=wd,
                                            patience=8, dev=dev, ls_eps=ls_eps, grad_clip=grad_clip)

                # (Optional) save weights – disabled by default for storage
                # torch.save(model.state_dict(), weight_path)

                # Eval: hidden codes -> cosine similarity -> distance-based accuracy buckets
                H = hidden_activations(model, n_items, dev)
                sims = cosine_sim(H, H)
                acc = relative_distance_accuracy(n_items, sims, D)

                # Pack results (same schema)
                results['name'].append(model_name)
                results['path'].append(weight_path)
                results['task'].append(dataset_ID)
                results['L2'].append(L2)
                results['1'].append(acc[1])
                results['2'].append(acc[2])
                results['3'].append(acc[3])
                results['4'].append(acc[4])
                results['scores'].append(acc)  # dict {1:..,2:..,3:..,4:..}
                results['end_loss'].append(best_loss)
                results['hidden'].append(H)     # latent codes
                results['dists'].append(sims)   # cosine similarity matrix
                results['learn_rate'].append(lr)
                results['weight_decay'].append(wd)
                results['repetition'].append(partition)     # kept same as your old script
                results['initial_weight_type'].append(iw)
                results['initial_weights'].append(initial_weights)

                # Minimal progress print (keeps logs readable on HPC)
                print(f"[{dataset_ID}] L2={L2:>3} rep={model_i:>3} seed={seed_i} "
                      f"| d1={acc[1]:.1f} d2={acc[2]:.1f} d3={acc[3]:.1f} d4={acc[4]:.1f}  loss={best_loss:.4f}",
                      flush=True)

    # Save results (same filename pattern)
    out_csv = f'output_data/{callback}_{partition}_{arrayid}.csv'
    r_frame = pd.DataFrame(results)
    r_frame.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
