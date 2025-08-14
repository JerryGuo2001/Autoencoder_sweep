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
from __future__ import annotations


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

# ------------------ schedules: one exposure per directed edge ------------------
def schedule_interleaved(adj: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    undirected = edges_from_adj(adj)
    directed = np.vstack([undirected, undirected[:, ::-1]])  # both directions once
    idx = np.arange(directed.shape[0]); rng.shuffle(idx)
    e = directed[idx]
    n = adj.shape[0]
    X = np.eye(n, dtype=np.float32)[e[:,0]]
    Y = e[:,1].astype(np.int64)
    return X, Y

def schedule_blocked_strict(adj: np.ndarray, nlists: int = 4, seed: int = 123,
                            exposures_per_edge: int = 1, epoch_index: int = 0):
    """
    Blocked schedule: present only ONE block this epoch, all its directed edges
    (optionally repeated), contiguous and without reshuffle.
    Blocks are fixed by `seed` and we cycle: epoch 0 -> block 0, epoch 1 -> block 1, etc.
    """
    undirected = edges_from_adj(adj)
    blocks = find_blocks_once(adj, nlists=nlists, seed=seed)  # fixed partition
    b = blocks[epoch_index % nlists]                          # pick block for this epoch
    sub = undirected[b]
    both = np.vstack([sub, sub[:, ::-1]])                     # both directions
    if exposures_per_edge > 1:
        both = np.repeat(both, exposures_per_edge, axis=0)    # consecutive repeats keep contiguity

    n = adj.shape[0]
    X = np.eye(n, dtype=np.float32)[both[:, 0]]
    Y = both[:, 1].astype(np.int64)
    return X, Y


class MinimalANN(nn.Module):
    def __init__(self, n_items=12, L1=12, L2=36, E3=12, init="he",
                 dropout_p: float = 0.3, input_dropout_p: float = 0.0):
        super().__init__()
        self.in_drop = nn.Dropout(p=input_dropout_p)
        self.E1 = nn.Linear(n_items, L1)
        self.E2 = nn.Linear(L1, L2)
        self.drop = nn.Dropout(p=dropout_p)     # <— dropout right after E2
        self.E3 = nn.Linear(L2, E3)
        self.out = nn.Linear(E3, n_items)
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
        # dropout only active in model.train()
        x = self.in_drop(x)
        x = F.relu(self.E1(x))
        x = F.relu(self.E2(x))
        x = self.drop(x)        # <— here
        h = self.E3(x)          # linear internal layer
        return h


    def forward(self, x):
        h = self.encode(x)
        logits = self.out(h)   # logits for CE
        return logits, h

# ------------------ training (essentials only) ------------------
def train_epoch_stream(model, X_np, Y_np, dev, opt, ce):
    X = torch.from_numpy(X_np).to(dev)
    Y = torch.from_numpy(Y_np).to(dev)
    last_loss = 0.0
    model.train()
    for i in range(X.shape[0]):  # batch size = 1
        logits, _ = model(X[i:i+1])
        loss = ce(logits, Y[i:i+1])
        opt.zero_grad(); loss.backward(); opt.step()
        last_loss = float(loss.item())
    return last_loss

def train(model, adj, regime, seed, epochs, lr, wd, nlists=4, exposures_per_edge=1):
    dev = device()
    model.to(dev)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)  # keep optimizer across epochs

    last = 0.0
    for ep in range(epochs):
        if regime == "I":
            # still reshuffle each epoch = truly interleaved
            X, Y = schedule_interleaved(adj, seed + ep*9973)
        else:
            # one block per epoch, no reshuffle, contiguous exposure
            X, Y = schedule_blocked_strict(adj, nlists=nlists, seed=seed, exposures_per_edge=exposures_per_edge, epoch_index=ep)

        last = train_epoch_stream(model, X, Y, dev, opt, ce)
    return last


# ------------------ evaluation (activations-only) ------------------
def hidden_activations(model, n_items, dev):
    I = torch.eye(n_items, dtype=torch.float32, device=dev)
    model.eval()
    with torch.no_grad():
        H = model.encode(I)
    return H.detach().cpu().numpy()

# ----- local-search distance using cosine over activations (beam) + softmax choice -----
def softmax_choice_from_hops(h12: float, h13: float, temperature: float = 1.0,
                             sample: bool = False, rng: Optional[np.random.Generator] = None):
    """
    Convert hop counts into a softmax choice over candidates:
      probs ~ softmax( -hops / tau )
    Returns (pick_idx, probs_array). pick_idx = 0 => choose i2; 1 => choose i3.
    """
    vals = np.array([h12, h13], dtype=float)
    big = 1e6  # penalty for failures
    vals[np.isinf(vals)] = big

    tau = max(1e-8, float(temperature))
    logits = -vals / tau
    logits -= np.max(logits)  # numerical stability
    exps = np.exp(logits)
    probs = exps / (exps.sum() + 1e-12)

    if sample:
        if rng is None:
            rng = np.random.default_rng()
        pick = int(rng.choice([0, 1], p=probs))
    else:
        pick = int(np.argmax(probs))
    return pick, probs

def beam_hops(adj: np.ndarray, sims: np.ndarray, src: int, tgt: int,
              beam_width: int = 3, max_steps: int = 10, require_improve: bool = True) -> int:
    """
    Simplified greedy hop counter:
      - Start at src.
      - Repeatedly move to the neighbor with highest cosine similarity to the target (from sims).
      - Stop if we reach tgt or after max_steps (default 10).
      - If not reached within max_steps, return np.inf (caller can randomly choose an option).

    Notes:
      * beam_width and require_improve are ignored; kept only for API compatibility.
      * Uses activations-only similarity (sims) to guide moves.
    """
    if src == tgt:
        return 0

    current = src
    visited = {src}

    for steps in range(1, max_steps + 1):
        # neighbors of current node
        nbrs = np.where(adj[current] > 0)[0]
        if nbrs.size == 0:
            return np.inf  # dead end
        # prefer unvisited neighbors; if none, allow revisits to avoid hard dead-ends
        unvisited = [v for v in nbrs if v not in visited]
        candidates = unvisited if len(unvisited) > 0 else list(nbrs)

        # pick the neighbor with highest similarity to the target
        next_node = max(candidates, key=lambda v: sims[v, tgt])

        if next_node == tgt:
            return steps

        visited.add(next_node)
        current = next_node

    # not reached within max_steps
    return np.inf

from collections import defaultdict

def distance_accuracy(adj: np.ndarray, H: np.ndarray, D_true: np.ndarray,
                      beam_width: int = 3, require_improve: bool = True,
                      softmax_temperature: float = 1.0, sample_softmax: bool = False,
                      rng: Optional[np.random.Generator] = None,
                      eval_noise_std: float = 0.0,
                      max_dd: Optional[int] = 4):
    """
    Relative distance via similarity-guided navigation:
      • Skip triple if ANY distance in {D(i1,i2), D(i3,i2), D(i1,i3)} == 1.
      • If both searches fail, prediction is a random coin flip.
      • If distances tie, ground truth is a random coin flip (expected 50%).
      • Bucket by |D(i1,i2) - D(i1,i3)|; values > max_dd are binned into max_dd.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Optional noise in evaluation
    if eval_noise_std > 0.0:
        H = H + rng.normal(0.0, eval_noise_std, size=H.shape)

    sims = 1 - cdist(H, H, metric="cosine")
    n = adj.shape[0]
    buckets = defaultdict(list)

    for i1, i2, i3 in itertools.permutations(range(n), 3):
        # --- skip if any distance is exactly 1 ---
        if (D_true[i1, i2] == 1) or (D_true[i3, i2] == 1) or (D_true[i1, i3] == 1):
            continue

        # True hop distances
        d12 = D_true[i1, i2]
        d13 = D_true[i1, i3]

        # Bucket index (absolute diff)
        dd = int(abs(d12 - d13))
        if (max_dd is not None) and (dd > max_dd):
            dd = max_dd

        # Ground truth
        if np.isfinite(d12) and np.isfinite(d13) and (d12 != d13):
            gt_is_i2 = d12 < d13
        else:
            gt_is_i2 = bool(rng.integers(0, 2))  # tie/infinite => coin flip

        # Navigation hops
        h12 = beam_hops(adj, sims, i1, i2, beam_width=beam_width, max_steps=10, require_improve=require_improve)
        h13 = beam_hops(adj, sims, i1, i3, beam_width=beam_width, max_steps=10, require_improve=require_improve)

        # Prediction
        if np.isinf(h12) and np.isinf(h13):
            pred_is_i2 = bool(rng.integers(0, 2))
        else:
            pick, _ = softmax_choice_from_hops(h12, h13,
                                               temperature=softmax_temperature,
                                               sample=sample_softmax,
                                               rng=rng)
            pred_is_i2 = (pick == 0)

        buckets[dd].append(1 if pred_is_i2 == gt_is_i2 else 0)

    return {k: (100.0 * np.mean(v) if len(v) else np.nan) for k, v in buckets.items()}



# ------------------ experiment runner (minimal) ------------------
def run(adj, L2_grid, n_models=5, regime="I", seed=123, epochs=1, lr=2.78e-4, wd=1.62e-2,
        L1=12, E3=12, beam_width=3, require_improve=True,
        softmax_temperature: float = 1.0, sample_softmax: bool = False, eval_noise_std: float = 0.0,drop_out=0.1):
    set_seed(seed)
    dev = device()
    n_items = adj.shape[0]
    D_true = shortest_paths(adj)
    records = []

    for L2 in L2_grid:
        for rep in range(n_models):
            model = MinimalANN(
                n_items=n_items, L1=L1, L2=L2, E3=E3, init="he",
                dropout_p=0.3,          # try 0.2–0.5; 0.3 is a good start
                input_dropout_p= drop_out
            ).to(dev)

            last_loss = train(model, adj, regime, seed + 1000*rep + 13*L2, epochs=epochs, lr=lr, wd=wd)

            # Activations-only
            H = hidden_activations(model, n_items, dev)

            # Per-model RNG so stochastic softmax decisions vary by rep/L2 but are reproducible
            rng = np.random.default_rng(seed + 37*rep + 11*L2)

            acc = distance_accuracy(
                adj, H, D_true,
                beam_width=beam_width,
                require_improve=require_improve,
                softmax_temperature=softmax_temperature,
                sample_softmax=sample_softmax,
                rng=rng,
                eval_noise_std=eval_noise_std
            )

            # Ensure all buckets 1..4 exist
            acc_full = {1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan}
            acc_full.update(acc)

            records.append({
                "regime": regime, "L2": L2, "rep": rep,
                "acc_d1": acc_full[1], "acc_d2": acc_full[2],
                "acc_d3": acc_full[3], "acc_d4": acc_full[4],
                "final_loss": last_loss
            })

            print(f"[{regime}] L2={L2:>3} rep={rep} | "
                  f"d1={acc_full[1]:.1f} d2={acc_full[2]:.1f} "
                  f"d3={acc_full[3]:.1f} d4={acc_full[4]:.1f}  loss={last_loss:.4f}")

    return pd.DataFrame.from_records(records)


# ------------------ run on your graph ------------------
if __name__ == "__main__":
    import os
    base_seed = int.from_bytes(os.urandom(8), "little") % (2**31 - 1)
    print(f"[Run] base_seed = {base_seed}")
    set_seed(base_seed)

    # Example graph (12 nodes)
    Gedges =  np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], # 0
                        [1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.], # 1
                        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], # 2
                        [0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0.], # 3
                        [0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0.], # 4
                        [0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.], # 5
                        [0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0.], # 6
                        [0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.], # 7
                        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.], # 8
                        [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1.], # 9
                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1.], # 10
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]], dtype=np.float32)

    L2_grid = [6, 36, 108, 216, 324]  # width of E2 (pattern separation capacity)

    # Minimal hyperparams (paper-inspired) + softmax choice controls
    common = dict(
        n_models=1000,
        epochs=1,              # adjust as needed
        lr=np.random.rand(),
        wd=np.random.rand(),
        L1=12, E3=12,
        beam_width=1,              # use the argument you set here
        require_improve=True,
        softmax_temperature=10,   # softer choices
        eval_noise_std=2,
        sample_softmax=True,       # stochastic decisions
        seed=base_seed,
        drop_out=0.5
    )

    df_B = run(Gedges, L2_grid, regime="B", **common)
    df_I = run(Gedges, L2_grid, regime="I", **common)

    for d in (df_B, df_I):
        d["seed"] = base_seed
    df_all = pd.concat([df_B, df_I], ignore_index=True)

    print("\nSummary (both regimes combined; distance via local search on activations):")
    print(df_all.groupby(["regime","L2"])[["acc_d1","acc_d2","acc_d3","acc_d4","final_loss"]].mean().round(1))

    # save ONE file
    os.makedirs("output_data", exist_ok=True)
    out_path = os.path.join(
        "output_data",
        f"SeqCompare_seed{base_seed}_{partition}_{callback}.csv"
    )
    df_all.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


