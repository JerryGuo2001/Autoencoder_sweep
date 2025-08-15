from __future__ import annotations
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
import itertools, random
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import networkx as nx
from scipy.spatial.distance import cdist
import itertools, random
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import networkx as nx
from scipy.spatial.distance import cdist
from collections import defaultdict
import json

partition = int(sys.argv[1])
callback= int(sys.argv[2])

def set_seed(s=123):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    try: torch.cuda.manual_seed_all(s)
    except Exception: pass

def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    try:
        if torch.backends.mps.is_available(): return torch.device("mps")
    except Exception: pass
    return torch.device("cpu")

def edges_from_adj(adj: np.ndarray):
    return np.array(list(nx.from_numpy_array(adj).edges()), dtype=int)

def shortest_paths(adj: np.ndarray):
    return np.asarray(nx.floyd_warshall_numpy(nx.from_numpy_array(adj)), dtype=float)

# ------------------ blocked partition (matching per block) ------------------
def find_blocks_once(adj: np.ndarray, nlists: int = 4, tries: int = 5000, seed: int = 123) -> List[np.ndarray]:
    G = nx.from_numpy_array(adj)
    edges = np.array(list(G.edges()), dtype=int)
    rng = np.random.default_rng(seed)

    def try_one():
        used = [set() for _ in range(nlists)]
        blocks = [[] for _ in range(nlists)]
        order = np.arange(len(edges)); rng.shuffle(order)
        for ei in order:
            u, v = edges[ei]
            for c in range(nlists):
                if (u not in used[c]) and (v not in used[c]):
                    blocks[c].append(ei); used[c].update((u, v)); break
            else:
                return None
        return [np.array(b, dtype=int) for b in blocks]

    for _ in range(tries):
        out = try_one()
        if out is not None: return out
    raise RuntimeError("Could not find a block partition; try raising nlists/tries.")

# ------------------ schedules ------------------
def schedule_interleaved(adj: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    undirected = edges_from_adj(adj)
    directed = np.vstack([undirected, undirected[:, ::-1]])
    idx = np.arange(directed.shape[0]); rng.shuffle(idx)
    e = directed[idx]
    n = adj.shape[0]
    X = np.eye(n, dtype=np.float32)[e[:,0]]
    Y = e[:,1].astype(np.int64)
    return X, Y

def schedule_blocked_strict(adj: np.ndarray, nlists: int = 4, seed: int = 123,
                            exposures_per_edge: int = 1, epoch_index: int = 0):
    undirected = edges_from_adj(adj)
    blocks = find_blocks_once(adj, nlists=nlists, seed=seed)
    b = blocks[epoch_index % nlists]
    sub = undirected[b]
    both = np.vstack([sub, sub[:, ::-1]])
    if exposures_per_edge > 1:
        both = np.repeat(both, exposures_per_edge, axis=0)
    n = adj.shape[0]
    X = np.eye(n, dtype=np.float32)[both[:, 0]]
    Y = both[:, 1].astype(np.int64)
    return X, Y

# ------------------ model: 5-layer symmetric with linear bottleneck ------------------
class MinimalANN(nn.Module):
    """
    input (n_items) -> L1 (12) -> L2 (grid) -> Bottleneck (B) -> L3 (=L2) -> L4 (=12) -> output (n_items)
    Bottleneck is linear and returned by encode() for the judgment task.
    """
    def __init__(self, n_items=12, L1=12, L2=36, B=12, init="he",
                 dropout_p: float = 0.3, input_dropout_p: float = 0.0):
        super().__init__()
        assert L1 == 12, "Per spec, L1 must be fixed at 12."
        self.in_drop = nn.Dropout(p=input_dropout_p)
        # encoder side
        self.L1 = nn.Linear(n_items, L1)
        self.L2 = nn.Linear(L1, L2)
        self.drop = nn.Dropout(p=dropout_p)     # dropout after L2
        self.B   = nn.Linear(L2, B)             # linear bottleneck (hidden activation H)
        # decoder side (symmetric)
        self.L3 = nn.Linear(B, L2)
        self.L4 = nn.Linear(L2, L1)
        self.out = nn.Linear(L1, n_items)
        self._init(init)

    def _init(self, how: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if how == "he":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif how == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif how == "lecun":
                    nn.init.normal_(m.weight, 0, (1 / m.weight.size(1))**0.5)
                elif how == "orthogonal":
                    nn.init.orthogonal_(m.weight)
                else:
                    raise ValueError(how)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        # Return linear bottleneck activations H (no ReLU here)
        x = self.in_drop(x)
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = self.drop(x)
        h = self.B(x)    # linear bottleneck (used for judgment)
        return h

    def forward(self, x):
        h = self.encode(x)
        # decode path, mirror sizes of L2 and L1
        x = F.relu(self.L3(h))
        x = F.relu(self.L4(x))
        logits = self.out(x)   # logits for CE
        return logits, h

# ------------------ training ------------------
def train_epoch_stream(model, X_np, Y_np, dev, opt, ce):
    X = torch.from_numpy(X_np).to(dev)
    Y = torch.from_numpy(Y_np).to(dev)
    last_loss = 0.0
    model.train()
    for i in range(X.shape[0]):
        logits, _ = model(X[i:i+1])
        loss = ce(logits, Y[i:i+1])
        opt.zero_grad(); loss.backward(); opt.step()
        last_loss = float(loss.item())
    return last_loss

def train(model, adj, regime, seed, epochs, lr, wd, nlists=4, exposures_per_edge=1):
    dev = device(); model.to(dev)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    last = 0.0
    for ep in range(epochs):
        if regime == "I":
            X, Y = schedule_interleaved(adj, seed + ep*9973)
        else:
            X, Y = schedule_blocked_strict(adj, nlists=nlists, seed=seed,
                                           exposures_per_edge=exposures_per_edge, epoch_index=ep)
        last = train_epoch_stream(model, X, Y, dev, opt, ce)
    return last

# ------------------ evaluation helpers ------------------
def hidden_activations(model, n_items, dev):
    I = torch.eye(n_items, dtype=torch.float32, device=dev)
    model.eval()
    with torch.no_grad():
        H = model.encode(I)   # <- bottleneck activations
    return H.detach().cpu().numpy()

def softmax_choice_from_hops(h12: float, h13: float, temperature: float = 1.0,
                             sample: bool = False, rng: Optional[np.random.Generator] = None):
    vals = np.array([h12, h13], dtype=float)
    big = 1e6; vals[np.isinf(vals)] = big
    tau = max(1e-8, float(temperature))
    logits = -vals / tau; logits -= np.max(logits)
    exps = np.exp(logits); probs = exps / (exps.sum() + 1e-12)
    if sample:
        if rng is None: rng = np.random.default_rng()
        pick = int(rng.choice([0, 1], p=probs))
    else:
        pick = int(np.argmax(probs))
    return pick, probs

# -------- uniform noise: edge dropout on the search graph (affects both regimes equally) --------
def drop_edges_undirected(adj: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    if p <= 0.0: return adj
    A = adj.copy()
    n = A.shape[0]
    iu, ju = np.triu_indices(n, 1)
    mask_edges = (A[iu, ju] > 0)
    idx = np.where(mask_edges)[0]
    if idx.size == 0: return A
    drop = rng.random(idx.size) < p
    di = iu[idx[drop]]; dj = ju[idx[drop]]
    A[di, dj] = 0.0; A[dj, di] = 0.0
    return A

# -------- TRUE beam-search evaluator (identical settings for B and I) --------
def beam_search_hops(adj: np.ndarray, sims: np.ndarray, src: int, tgt: int,
                     beam_width: int = 2, max_steps: int = 8,
                     eps: float = 0.05,
                     rng: Optional[np.random.Generator] = None) -> int:
    if src == tgt: return 0
    if rng is None: rng = np.random.default_rng()
    frontier = np.array([src], dtype=int)
    for steps in range(1, max_steps + 1):
        cand = []
        for u in frontier:
            nbrs = np.where(adj[u] > 0)[0]
            if nbrs.size == 0: continue
            scores = sims[nbrs, tgt].astype(float)
            order = np.argsort(-scores)
            if (order.size > 1) and (rng.random() < eps):
                order[0], order[1] = order[1], order[0]
            take = order[:beam_width] if order.size > beam_width else order
            cand.extend(nbrs[take].tolist())
        if not cand: return np.inf
        cand = np.array(cand, dtype=int)
        if (cand == tgt).any(): return steps
        uniq = np.unique(cand)
        best_order = np.argsort(-sims[uniq, tgt])
        pick = best_order[:beam_width] if best_order.size > beam_width else best_order
        frontier = uniq[pick]
    return np.inf

def distance_accuracy_beam(adj_eval: np.ndarray, H: np.ndarray, D_true: np.ndarray,
                           beam_width: int = 2, max_steps: int = 8,
                           eps: float = 0.05,
                           softmax_temperature: float = 1.0,
                           sample_softmax: bool = True,
                           rng: Optional[np.random.Generator] = None,
                           eval_noise_std: float = 0.0,
                           max_dd: Optional[int] = 4):
    if rng is None: rng = np.random.default_rng()
    Hc = H.copy()
    if eval_noise_std > 0.0:
        Hc = Hc + rng.normal(0.0, eval_noise_std, size=Hc.shape)

    sims = 1 - cdist(Hc, Hc, metric="cosine")
    n = adj_eval.shape[0]
    buckets = defaultdict(list)

    for i1, i2, i3 in itertools.permutations(range(n), 3):
        if (D_true[i1, i2] == 1) or (D_true[i3, i2] == 1) or (D_true[i1, i3] == 1):
            continue
        d12, d13 = D_true[i1, i2], D_true[i1, i3]
        dd = int(abs(d12 - d13)); dd = min(dd, max_dd) if max_dd is not None else dd

        if np.isfinite(d12) and np.isfinite(d13) and (d12 != d13):
            gt_is_i2 = d12 < d13
        else:
            gt_is_i2 = bool(rng.integers(0, 2))

        h12 = beam_search_hops(adj_eval, sims, i1, i2, beam_width=beam_width, max_steps=max_steps, eps=eps, rng=rng)
        h13 = beam_search_hops(adj_eval, sims, i1, i3, beam_width=beam_width, max_steps=max_steps, eps=eps, rng=rng)

        if np.isinf(h12) and np.isinf(h13):
            pred_is_i2 = bool(rng.integers(0, 2))
        else:
            pick, _ = softmax_choice_from_hops(h12, h13, temperature=softmax_temperature,
                                               sample=sample_softmax, rng=rng)
            pred_is_i2 = (pick == 0)

        buckets[dd].append(1 if pred_is_i2 == gt_is_i2 else 0)

    return {k: (100.0 * np.mean(v) if len(v) else np.nan) for k, v in buckets.items()}

# ------------------ experiment runner (uniform evaluator for B & I) ------------------
def run_uniform_eval(adj, L2_grid, n_models=5, regime="I", seed=123, epochs=1, lr=2.78e-4, wd=1.62e-2,
                     bottleneck_dim: int = 3, drop_out: float = 0.1,
                     # evaluator settings (IDENTICAL for B & I)
                     beam_width: int = 2, eps: float = 0.05, max_steps: int = 8,
                     softmax_temperature: float = 1.0, sample_softmax: bool = True,
                     # uniform difficulty knobs (apply to BOTH regimes)
                     eval_noise_std: float = 0.3, edge_dropout_p: float = 0.08):
    """
    Training unchanged; evaluation uses true beam-search with identical settings for B and I.
    Applies uniform activation noise AND edge dropout on the search graph.
    Prints ONE line per model with its overall accuracy; returns a DataFrame.
    """
    set_seed(seed)
    dev = device()
    n_items = adj.shape[0]
    D_true = shortest_paths(adj)
    records = []

    eede = (f"beam={beam_width};eps={eps};max_steps={max_steps};tau={softmax_temperature};"
            f"eval_noise_std={eval_noise_std};edge_drop={edge_dropout_p};B={bottleneck_dim}")

    for L2w in L2_grid:
        for rep in range(n_models):
            model = MinimalANN(
                n_items=n_items, L1=12, L2=L2w, B=bottleneck_dim, init="he",
                dropout_p=0.3, input_dropout_p=drop_out
            ).to(dev)

            # ---- train
            last_loss = train(model, adj, regime, seed + 1000*rep + 13*L2w, epochs=epochs, lr=lr, wd=wd)

            # ---- eval
            H = hidden_activations(model, n_items, dev)  # bottleneck activations
            rng = np.random.default_rng(seed + 37*rep + 11*L2w)

            # uniform edge dropout on the EVAL graph (same for both regimes)
            adj_eval = drop_edges_undirected(adj, edge_dropout_p, rng)

            acc = distance_accuracy_beam(
                adj_eval, H, D_true,
                beam_width=beam_width, max_steps=max_steps, eps=eps,
                softmax_temperature=softmax_temperature, sample_softmax=sample_softmax,
                rng=rng, eval_noise_std=eval_noise_std
            )

            # buckets → per-model overall
            acc_full = {1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan}
            acc_full.update(acc)
            acc_overall = float(np.nanmean([acc_full[1], acc_full[2], acc_full[3], acc_full[4]]))

            # one compact row per model
            print(f"[{regime}] L2={L2w:>3} rep={rep:>2} | overall={acc_overall:5.2f}")

            records.append({
                "regime": regime, "L2": L2w, "rep": rep,
                "acc_d1": acc_full[1], "acc_d2": acc_full[2],
                "acc_d3": acc_full[3], "acc_d4": acc_full[4],
                "final_loss": last_loss, "eede": eede, "acc_overall": acc_overall
            })

    return pd.DataFrame.from_records(records)

# ------------------ run on your graph (final df + prints) ------------------
if __name__ == "__main__":
    import os
    base_seed = int.from_bytes(os.urandom(8), "little") % (2**31 - 1)
    set_seed(base_seed)

    # Graph
    Gedges =  np.array([[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                        [1.,0.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.],
                        [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
                        [0.,1.,0.,0.,1.,0.,1.,1.,0.,0.,0.,0.],
                        [0.,1.,1.,1.,0.,1.,0.,0.,0.,0.,0.,0.],
                        [0.,0.,0.,0.,1.,0.,0.,1.,0.,0.,0.,0.],
                        [0.,0.,0.,1.,0.,0.,0.,0.,1.,1.,0.,0.],
                        [0.,0.,0.,1.,0.,1.,0.,0.,0.,1.,0.,0.],
                        [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.],
                        [0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,1.,1.],
                        [0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,1.],
                        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.]], dtype=np.float32)

    # L2 width grid (pattern separation capacity in the first expansion)
    L2_grid = [6,12, 24, 36]

    common = dict(
        n_models=10, epochs=5, lr=0.388731, wd=0.138503,
        seed=base_seed, drop_out=0.5,
        # evaluator (identical for B & I)
        beam_width=1, eps=0.15, max_steps=3, softmax_temperature=2.0, sample_softmax=True,
        # uniform difficulty:
        eval_noise_std=0.0, edge_dropout_p=0.0,
        # bottleneck width
        bottleneck_dim=12
    )

    df_B = run_uniform_eval(Gedges, L2_grid, regime="B", **common)
    df_I = run_uniform_eval(Gedges, L2_grid, regime="I", **common)

    for d in (df_B, df_I):
        d["seed"] = base_seed
    df = pd.concat([df_B, df_I], ignore_index=True)

    # save ONE file
    os.makedirs("output_data", exist_ok=True)
    out_path = os.path.join(
        "output_data",
        f"SeqCompare_seed{base_seed}_{partition}_{callback}.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
