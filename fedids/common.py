"""Shared model, data and partition utilities."""
import functools

import numpy as np
import torch
from torch import nn

CHECKPOINT_ROUNDS = (10, 50, 100)
TRAJECTORY_EVERY = 10


class MLP(nn.Module):
    def __init__(self, d_in, n_cls, h1=128, h2=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(),
                                 nn.Linear(h2, n_cls))

    def forward(self, x):
        return self.net(x)


@functools.lru_cache(maxsize=4)
def load_pool(path):
    d = np.load(path)
    return d["X"], d["y"], d["split"]


@functools.lru_cache(maxsize=8)
def load_partition(path):
    d = np.load(path)
    return d["client"]


def dirichlet_partition(y_members, n_clients, alpha, seed, min_size=50):
    """Label-skew partition: for each class, split its members across clients by Dir(alpha)."""
    rng = np.random.default_rng(seed)
    n_cls = int(y_members.max()) + 1
    for _ in range(1000):
        client = np.empty(len(y_members), dtype=np.int64)
        for c in range(n_cls):
            idx = rng.permutation(np.flatnonzero(y_members == c))
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx, cuts)):
                client[part] = k
        if np.bincount(client, minlength=n_clients).min() >= min_size:
            return client
    raise RuntimeError("could not satisfy min_size")


def member_indices(split):
    return np.flatnonzero(split == "member")


def client_data(pool_path, part_path, pid):
    X, y, split = load_pool(pool_path)
    mem = member_indices(split)
    client = load_partition(part_path)
    idx = mem[client == pid]
    return X[idx], y[idx]


def state_to_vec(sd):
    return torch.cat([v.reshape(-1).float() for v in sd.values()])


def build_model(pool_path):
    X, y, _ = load_pool(pool_path)
    return MLP(X.shape[1], int(y.max()) + 1)
