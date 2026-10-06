"""Canary pool for the positive control.

Copies a study pool and flips the label of C member records and C non-member records to a
random wrong class. The member canaries are trained on by their clients; the non-member
canaries are never seen. The Dirichlet partition of the original pool is reused, so the
federation differs from the study pool only in the C flipped labels.
"""
import argparse
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.expanduser("~/fedids_work"))
    ap.add_argument("--dataset", default="cicids2017")
    ap.add_argument("--alphas", default="0.1,0.5,1.0")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    d = np.load(os.path.join(a.work, "data", f"{a.dataset}.npz"))
    X, y, split = d["X"], d["y"].copy(), d["split"]
    rng = np.random.default_rng(a.seed)
    n_cls = int(y.max()) + 1
    mem = rng.choice(np.flatnonzero(split == "member"), a.n, replace=False)
    non = rng.choice(np.flatnonzero(split == "nonmember"), a.n, replace=False)
    idx = np.concatenate([mem, non])
    y[idx] = (y[idx] + 1 + rng.integers(0, n_cls - 1, len(idx))) % n_cls
    name = f"{a.dataset}_canary"
    np.savez_compressed(os.path.join(a.work, "data", f"{name}.npz"), X=X, y=y, split=split,
                        canary_idx=idx, canary_is_member=np.r_[np.ones(a.n, np.int8), np.zeros(a.n, np.int8)])
    for alpha in a.alphas.split(","):
        src = os.path.join(a.work, "partitions", f"{a.dataset}_a{float(alpha)}_k10.npz")
        shutil.copy(src, os.path.join(a.work, "partitions", f"{name}_a{float(alpha)}_k10.npz"))
    print(name, len(idx))


if __name__ == "__main__":
    main()
