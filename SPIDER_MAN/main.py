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

def edges_from_adj(adj: np.ndarray):
    return np.array(list(nx.from_numpy_array(adj).edges()), dtype=int)

def shortest_paths(adj: np.ndarray):
    return np.asarray(nx.floyd_warshall_numpy(nx.from_numpy_array(adj)), dtype=float)

def cosine_sim(a, b):  # for evaluation only
    return 1 - cdist(a, b, metric="cosine")

# ------------------ blocked partition (matching per block) ------------------
def find_blocks_once(adj: np.ndarray, nlists: int = 4, tries: int = 10000, seed: int = 123) -> List[np.ndarray]:
    G = nx.from_numpy_array(adj)
    edges = np.array(list(G.edges()), dtype=int)
    nE = len(edges); rng = np.random.default_rng(seed)

    def try_one():
        used = [set() for _ in range(nlists)]
        blocks = [[] for _ in range(nlists)]
        order = np.arange(nE); rng.shuffle(order)
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
    raise RuntimeError("Could not find a block partition; raise nlists/tries.")

# ------------------ sequence builder (equal exposures) ------------------
def make_sequence_equalized(adj: np.ndarray, regime: str, seed: int,
                            exposures_per_edge: int = 20, nlists: int = 4,
                            bidirectional: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    undirected_edges = edges_from_adj(adj)                 # shape (E, 2)
    if bidirectional:
        # duplicate both directions but keep the SAME total exposures per undirected edge
        # so each direction gets exposures_per_edge // 2 (±1 for remainder)
        edges = np.vstack([undirected_edges, undirected_edges[:, ::-1]])
        # when we assign repeats below, we’ll split evenly across the two halves
        split = True
    else:
        edges = undirected_edges
        split = False

    n_items = adj.shape[0]
    eye = np.eye(n_items, dtype=np.float32)

    if regime == "I":
        if split:
            per_dir = exposures_per_edge // 2
            rem = exposures_per_edge % 2
            # build index list for the two halves
            e1 = edges[:len(edges)//2]; e2 = edges[len(edges)//2:]
            idx1 = np.repeat(np.arange(len(e1)), per_dir + (1 if rem > 0 else 0))
            idx2 = np.repeat(np.arange(len(e2)), per_dir)
            rng.shuffle(idx1); rng.shuffle(idx2)
            e = np.vstack([e1[idx1], e2[idx2]])
            rng.shuffle(e)
        else:
            idx = np.repeat(np.arange(len(edges)), exposures_per_edge)
            rng.shuffle(idx)
            e = edges[idx]
        return eye[e[:,0]], e[:,1].astype(np.int64)

    if regime == "B":
        blocks = find_blocks_once(adj, nlists=nlists, seed=seed)
        # split total exposures per undirected edge across blocks and directions
        if split:
            per_dir = exposures_per_edge // 2
            rem = exposures_per_edge % 2
        else:
            per_dir = exposures_per_edge
            rem = 0

        Xs, Ys = [], []
        for bi, block in enumerate(blocks):
            # exposures for this block
            per_block = per_dir // len(blocks)
            # give the remainder to early blocks
            reps_this_block = per_block + (1 if bi < (per_dir % len(blocks)) else 0)
            if reps_this_block == 0 or len(block) == 0:
                continue

            sub_ud = undirected_edges[block]  # (k,2)
            if split:
                # forward in this block
                idx = np.repeat(np.arange(len(sub_ud)), reps_this_block + (1 if rem and bi == 0 else 0))
                rng.shuffle(idx)
                e_fwd = sub_ud[idx]
                Xs.append(eye[e_fwd[:,0]]); Ys.append(e_fwd[:,1].astype(np.int64))
                # backward in this block
                idx = np.repeat(np.arange(len(sub_ud)), reps_this_block)
                rng.shuffle(idx)
                e_bwd = sub_ud[:, ::-1][idx]
                Xs.append(eye[e_bwd[:,0]]); Ys.append(e_bwd[:,1].astype(np.int64))
            else:
                idx = np.repeat(np.arange(len(sub_ud)), reps_this_block)
                rng.shuffle(idx)
                e = sub_ud[idx]
                Xs.append(eye[e[:,0]]); Ys.append(e[:,1].astype(np.int64))

        if not Xs:
            return make_sequence_equalized(adj, "I", seed, exposures_per_edge, nlists, bidirectional=bidirectional)
        X = np.vstack(Xs); Y = np.concatenate(Ys)
        return X, Y

    raise ValueError("regime must be 'I' or 'B'")


# ------------------ model: tiny MLP + linear classifier (plain CE) ------------------
class SimpleCE(nn.Module):
    def __init__(self, n_items, L1=12, L2=36, n_hidden=12, p_drop=0.0, init="he"):
        super().__init__()
        self.E1 = nn.Linear(n_items, L1)
        self.E2 = nn.Linear(L1, L2)
        self.E3 = nn.Linear(L2, n_hidden)
        self.drop = nn.Dropout(p_drop)
        self.out = nn.Linear(n_hidden, n_items)
        self._init(init)

    def _init(self, how):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if how=="xavier": nn.init.xavier_uniform_(m.weight)
                elif how=="he": nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif how=="lecun": nn.init.normal_(m.weight, 0, (1/m.weight.size(1))**0.5)
                elif how=="orthogonal": nn.init.orthogonal_(m.weight)
                else: raise ValueError(how)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def encode(self, x):
        x = F.relu(self.E1(x))
        x = self.drop(F.relu(self.E2(x)))
        return self.E3(x)  # linear hidden

    def forward(self, x):
        h = self.encode(x)
        return self.out(h), h  # logits, hidden

# ------------------ training: per-trial CE (simple) ------------------
def train_ce(model, X_np, Y_np, epochs=50, lr=5e-4, wd=1e-4, dev=None, label_smoothing=0.02):
    dev = dev or device()
    X = torch.from_numpy(X_np).to(dev)
    Y = torch.from_numpy(Y_np).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for _ in range(epochs):
        model.train()
        for i in range(X.shape[0]):  # strict online updates
            logits, _ = model(X[i:i+1])
            loss = ce(logits, Y[i:i+1])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return float(loss.item())

# ------------------ evaluation: relative distance accuracy ------------------
def hidden_activations(model, n_items, dev):
    I = torch.eye(n_items, dtype=torch.float32, device=dev)
    model.eval()
    with torch.no_grad():
        _, H = model(I)
    return H.detach().cpu().numpy()

def relative_distance_accuracy(n_items, sims, D):
    buckets = {1:[], 2:[], 3:[], 4:[]}
    for i1, i2, i3 in itertools.permutations(range(n_items), 3):
        if D[i1,i2] <= 1 or D[i3,i2] <= 1 or D[i1,i3] <= 1:  # exclude identical/direct
            continue
        dd = int(abs(D[i1,i2] - D[i3,i2]))
        if dd not in buckets: continue
        correct = 0 if D[i1,i2] < D[i3,i2] else 1
        model_choice = 0 if sims[i1,i2] > sims[i3,i2] else 1
        buckets[dd].append(1 if model_choice==correct else 0)
    return {k: (100*np.mean(v) if len(v) else np.nan) for k,v in buckets.items()}

# ------------------ experiment runner ------------------
def run(adj, L2_grid, n_models=5, regime="I", seed=123,
        L1=12, n_hidden=12, p_drop=0.0, lr=2.5e-4, wd=5e-4,
        epochs=50, exposures_per_edge=20, nlists=4):
    set_seed(seed)
    dev = device()
    n_items = adj.shape[0]
    D = shortest_paths(adj)
    records = []

    for L2 in L2_grid:
        for rep in range(n_models):
            set_seed(seed + 31*rep + 7*L2)
            X, Y = make_sequence_equalized(adj, regime, seed + 1000*rep + 13*L2,
                                           exposures_per_edge=exposures_per_edge, nlists=nlists)
            model = SimpleCE(n_items, L1=L1, L2=L2, n_hidden=n_hidden, p_drop=p_drop, init="he").to(dev)
            last_loss = train_ce(model, X, Y, epochs=epochs, lr=lr, wd=wd, dev=dev,label_smoothing=0.01)

            H = hidden_activations(model, n_items, dev)
            Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
            sims = cosine_sim(Hn, Hn)
            acc = relative_distance_accuracy(n_items, sims, D)

            records.append({"regime":regime, "L2":L2, "rep":rep,
                            "acc_d1":acc[1], "acc_d2":acc[2], "acc_d3":acc[3], "acc_d4":acc[4],
                            "final_loss": last_loss})
            print(f"[{regime}] L2={L2:>3} rep={rep} | d1={acc[1]:.1f} d2={acc[2]:.1f} d3={acc[3]:.1f} d4={acc[4]:.1f}  loss={last_loss:.4f}")

    return pd.DataFrame.from_records(records)


# ------------------ run on your graph ------------------
if __name__ == "__main__":
    import os

    # fresh random seed each run, shared across B vs I
    base_seed = int.from_bytes(os.urandom(8), "little") % (2**31 - 1)
    print(f"[Run] base_seed = {base_seed}")
    set_seed(base_seed)

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

    # sweep
    L2_grid = [6, 36, 108, 216, 324]

    # shared hyperparams (keep your current choices)
    common = dict(
        n_models=100,
        L1=12,
        n_hidden=8,
        p_drop=0.0,
        lr=2.5e-4,
        wd=5e-4,
        epochs=1,
        exposures_per_edge=20,
        nlists=4,
        seed=base_seed,
    )

    # run both regimes with the same seed
    df_B = run(Gedges, L2_grid, regime="B", **common)
    df_I = run(Gedges, L2_grid, regime="I", **common)

    # combine and tag with seed
    for d in (df_B, df_I):
        d["seed"] = base_seed
    df_all = pd.concat([df_B, df_I], ignore_index=True)

    # summaries
    print("\nSummary (both regimes combined):")
    print(df_all.groupby(["regime","L2"])[["acc_d1","acc_d2","acc_d3","acc_d4","final_loss"]].mean().round(1))

    # save ONE file
    os.makedirs("output_data", exist_ok=True)
    out_path = os.path.join("output_data", f"SeqCompare_seed{base_seed}_{partition}_{callback}.csv")
    df_all.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

