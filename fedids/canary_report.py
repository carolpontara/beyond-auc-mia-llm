"""Positive control: exposure of label-flipped canaries under the same audit.

Canary TPR is reported twice: with the threshold calibrated on the regular non-members, as in
the main audit, and with the threshold calibrated on the 200 non-member canaries.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analyze import RATES, parse, threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.expanduser("~/fedids_work"))
    ap.add_argument("--out", default="results/canary.csv")
    a = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(a.work, "scores", "*_canary", "*", "*", "*.npz"))):
        ds, alpha, agg, seed = parse(path, a.work)
        pool = np.load(os.path.join(a.work, "data", f"{ds}.npz"))
        cidx, cmem = pool["canary_idx"], pool["canary_is_member"].astype(bool)
        z = np.load(path)
        ev, mem = z["eval_idx"], z["is_member"].astype(bool)
        pos = {i: k for k, i in enumerate(ev)}
        ci = np.array([pos[i] for i in cidx])
        regular_non = ~mem
        regular_non[ci[~cmem]] = False
        for att in [k for k in z.files if "@" in k]:
            s = z[att].astype(np.float64)
            name, r = att.split("@")
            row = {"dataset": ds, "alpha": alpha, "agg": agg, "seed": seed, "attack": name, "round": int(r),
                   "canary_auc": roc_auc_score(cmem, s[ci])}
            for q in RATES:
                h = threshold(s[regular_non], q)
                row[f"canary_tpr@{q:g}"] = float((s[ci[cmem]] >= h).mean())
                row[f"canary_fpr@{q:g}"] = float((s[ci[~cmem]] >= h).mean())
                row[f"regular_tpr@{q:g}"] = float((s[mem & ~np.isin(np.arange(len(s)), ci)] >= h).mean())
                hc = threshold(s[ci[~cmem]], q)
                row[f"canary_tpr_canarycal@{q:g}"] = float((s[ci[cmem]] >= hc).mean())
            rows.append(row)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    cols = ["canary_auc", "canary_tpr@0.05", "canary_fpr@0.05", "canary_tpr_canarycal@0.05"]
    print(pd.DataFrame(rows).groupby(["agg", "attack", "round"])[cols].mean().round(3))


if __name__ == "__main__":
    main()
