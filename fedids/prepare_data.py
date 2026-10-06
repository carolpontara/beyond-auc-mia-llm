"""Build the fixed study pools for CIC-IDS2017 and CICIoT2023.

Each pool is split once, stratified by class, into four disjoint sets:
members (distributed to the clients), non-members (MIA evaluation),
reference (attacker population data) and test (detection utility).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CIC17_GROUPS = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS", "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "Bot": "Bot",
}
CLIP = 10.0
SPLIT_FRACTIONS = {"member": 0.40, "nonmember": 0.20, "reference": 0.20, "test": 0.20}


def _clean(df):
    df.columns = [c.strip() for c in df.columns]
    feats = [c for c in df.columns if c != "cls"]
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan)
    X["cls"] = df["cls"].values
    return X.dropna().drop_duplicates()


def load_cicids2017(root, benign_per_file=60000, seed=42):
    rng = np.random.default_rng(seed)
    frames = []
    for f in sorted(glob.glob(os.path.join(root, "*.parquet"))):
        df = pd.read_parquet(f)
        df.columns = [c.strip() for c in df.columns]
        lab = df["Label"].astype(str)
        group = lab.map(CIC17_GROUPS)
        group[lab.str.startswith("Web Attack")] = "WebAttack"
        df = df.drop(columns=["Label"])
        df["cls"] = group.values
        df = df[df["cls"].notna()]
        ben = np.flatnonzero(df["cls"].to_numpy() == "Benign")
        drop = rng.permutation(ben)[benign_per_file:]
        df = df.drop(index=df.index[drop])
        frames.append(_clean(df))
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def load_ciciot2023(root):
    df = pd.read_parquet(os.path.join(root, "train.parquet"))
    df["cls"] = df["attack_class"].astype(str)
    return _clean(df.drop(columns=["Label", "attack_class", "label"]))


def build(df, cap_benign, cap_attack, seed):
    feats = [c for c in df.columns if c != "cls"]
    const = [c for c in feats if df[c].nunique() <= 1]
    both = df.drop(columns=const).drop_duplicates()
    rng = np.random.default_rng(seed)
    parts = []
    for c, g in both.groupby("cls"):
        cap = cap_benign if c in ("Benign",) else cap_attack
        idx = rng.permutation(len(g))[: min(cap, len(g))]
        parts.append(g.iloc[np.sort(idx)])
    pool = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    classes = sorted(pool["cls"].unique(), key=lambda c: (c != "Benign", c))
    y = pool["cls"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int64)
    Xp = pool.drop(columns=["cls"]).to_numpy(np.float64)
    Xp = np.sign(Xp) * np.log1p(np.abs(Xp))
    idx = np.arange(len(y))
    split = np.empty(len(y), dtype="<U10")
    rest = idx
    remaining = 1.0
    for name in ["member", "nonmember", "reference"]:
        frac = SPLIT_FRACTIONS[name] / remaining
        take, rest = train_test_split(rest, train_size=frac, stratify=y[rest], random_state=seed)
        split[take] = name
        remaining -= SPLIT_FRACTIONS[name]
    split[rest] = "test"
    ref = split == "reference"
    mu, sd = Xp[ref].mean(0), Xp[ref].std(0)
    ok = sd > 1e-3
    Xs = np.clip((Xp[:, ok] - mu[ok]) / sd[ok], -CLIP, CLIP).astype(np.float32)
    names = [c for c, k in zip(pool.drop(columns=["cls"]).columns, ok) if k]
    return Xs, y, split, classes, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cicids2017", "ciciot2023"], required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-benign", type=int, default=20000)
    ap.add_argument("--cap-attack", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = load_cicids2017(a.raw) if a.dataset == "cicids2017" else load_ciciot2023(a.raw)
    X, y, split, classes, feats = build(df, a.cap_benign, a.cap_attack, a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, X=X, y=y, split=split)
    summary = {
        "dataset": a.dataset, "n": int(len(y)), "features": len(feats), "classes": classes,
        "per_split_class": {s: np.bincount(y[split == s], minlength=len(classes)).tolist()
                            for s in SPLIT_FRACTIONS},
    }
    with open(a.out.replace(".npz", ".json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
