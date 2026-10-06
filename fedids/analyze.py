"""Aggregate MIA scores into the audit tables.

Thresholds are calibrated on the non-member set only: h(a) is the (1 - a) quantile of the
non-member scores, and a training record is "exposed" at rate a when its score exceeds h(a).
Because every training record of the federation is in the member set, exposure at rate a is
the fraction of the training data identified, i.e. TPR at FPR a.

Per class, the false positive rate of the same global threshold is reported next to the TPR.
Per client there are no client non-members, so the client exposure is compared with the
class-matched FPR, i.e. the non-member FPR of each class averaged over the client's records;
the difference is the exposure that the client's class mix alone does not explain.

Stability is the mean pairwise Jaccard index, across seeds, of the members flagged at 5% FPR,
reported with its chance level and with the same index computed on the flagged non-members:
a member set that is as stable as the non-member set reflects record difficulty, not
memorisation.
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import load_partition, load_pool, member_indices

RATES = (0.01, 0.05)
UNION_BUDGET = 0.01


def threshold(nonmember_scores, a):
    """Smallest h with FPR = mean(nonmember >= h) <= a; flag a record when score >= h."""
    v = np.sort(nonmember_scores)[::-1]
    k = int(np.floor(a * len(v)))
    if k == 0:
        return np.inf
    h = v[k - 1]
    while k > 0 and np.mean(nonmember_scores >= h) > a:
        k -= 1
        h = v[k - 1] if k > 0 else np.inf
    return float(h)


def run_metrics(s, mem):
    nm = s[~mem]
    row = {"auc": roc_auc_score(mem, s)}
    flagged = {}
    for a in RATES:
        h = threshold(nm, a)
        f = s >= h
        row[f"tpr@{a:g}"] = f[mem].mean()
        row[f"fpr@{a:g}"] = f[~mem].mean()
        flagged[a] = f
    return row, flagged


def parse(path, work):
    rel = os.path.relpath(path, os.path.join(work, "scores")).split(os.sep)
    return rel[0], float(rel[1][1:]), rel[2], int(rel[3][1:-4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.expanduser("~/fedids_work"))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs, per_class, per_client, unions, flags = [], [], [], [], {}
    for path in sorted(glob.glob(os.path.join(a.work, "scores", "*", "*", "*", "*.npz"))):
        ds, alpha, agg, seed = parse(path, a.work)
        if ds.endswith("_canary"):
            continue
        z = np.load(path)
        ev, mem = z["eval_idx"], z["is_member"].astype(bool)
        _, y, split = load_pool(os.path.join(a.work, "data", f"{ds}.npz"))
        part = load_partition(os.path.join(a.work, "partitions", f"{ds}_a{alpha}_k10.npz"))
        client_of = np.full(len(y), -1)
        client_of[member_indices(split)] = part
        yc, cc = y[ev], client_of[ev]
        attacks = [k for k in z.files if "@" in k]
        util = json.load(open(os.path.join(a.work, "runs", ds, f"a{alpha}", agg, f"s{seed}", "utility.json")))
        hist = {h["round"]: h for h in util["history"]}
        for att in attacks:
            s = z[att].astype(np.float64)
            name, r = att.split("@")
            row, fl = run_metrics(s, mem)
            runs.append({"dataset": ds, "alpha": alpha, "agg": agg, "seed": seed, "attack": name,
                         "round": int(r), "test_acc": hist[int(r)]["acc"],
                         "test_macro_f1": hist[int(r)]["macro_f1"], **row})
            flags[(ds, alpha, agg, seed, att)] = (ev[mem & fl[0.05]], ev[mem & fl[0.01]], ev[~mem & fl[0.05]])
            class_fpr = {}
            for c in np.unique(yc):
                m = yc == c
                class_fpr[c] = {q: fl[q][m & ~mem].mean() for q in RATES}
                per_class.append({"dataset": ds, "alpha": alpha, "agg": agg, "seed": seed, "attack": name,
                                  "round": int(r), "class": int(c),
                                  "auc": roc_auc_score(mem[m], s[m]),
                                  "tpr@0.01": fl[0.01][m & mem].mean(), "tpr@0.05": fl[0.05][m & mem].mean(),
                                  "fpr@0.01": class_fpr[c][0.01], "fpr@0.05": class_fpr[c][0.05]})
            for k in range(part.max() + 1):
                m = cc == k
                expected = {q: np.mean([class_fpr[c][q] for c in yc[m]]) for q in RATES}
                per_client.append({"dataset": ds, "alpha": alpha, "agg": agg, "seed": seed, "attack": name,
                                   "round": int(r), "client": k, "n_train": int(m.sum()),
                                   "tpr@0.01": fl[0.01][m].mean(), "tpr@0.05": fl[0.05][m].mean(),
                                   "class_matched_fpr@0.01": expected[0.01],
                                   "class_matched_fpr@0.05": expected[0.05]})
        final = [k for k in attacks if k.endswith("@100")]
        nm = ~mem
        desc = np.zeros(len(mem), bool)
        for k in final:
            desc |= z[k] >= threshold(z[k][nm], UNION_BUDGET)
        cal = np.zeros(len(mem), bool)
        for k in final:
            cal |= z[k] >= threshold(z[k][nm], UNION_BUDGET / len(final))
        unions.append({"dataset": ds, "alpha": alpha, "agg": agg, "seed": seed, "n_attacks": len(final),
                       "desc_tpr": desc[mem].mean(), "desc_fpr": desc[nm].mean(),
                       "cal_tpr": cal[mem].mean(), "cal_fpr": cal[nm].mean()})
    R = pd.DataFrame(runs)
    R.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    pd.DataFrame(per_class).to_csv(os.path.join(a.out, "per_class.csv"), index=False)
    pd.DataFrame(per_client).to_csv(os.path.join(a.out, "per_client.csv"), index=False)
    pd.DataFrame(unions).to_csv(os.path.join(a.out, "union.csv"), index=False)
    stab = []
    n_members = {ds: int((load_pool(os.path.join(a.work, "data", f"{ds}.npz"))[2] == "member").sum())
                 for ds in R.dataset.unique()}
    cells = R[["dataset", "alpha", "agg", "attack", "round"]].drop_duplicates().itertuples(index=False)
    for ds, alpha, agg, name, r in cells:
        seeds = sorted(R[(R.dataset == ds) & (R.alpha == alpha) & (R["agg"] == agg)].seed.unique())
        att = f"{name}@{r}"
        jac, chance, jac_nm = [], [], []
        for s1, s2 in itertools.combinations(seeds, 2):
            A, B = set(flags[(ds, alpha, agg, s1, att)][0]), set(flags[(ds, alpha, agg, s2, att)][0])
            if A | B:
                jac.append(len(A & B) / len(A | B))
            p, q = len(A) / n_members[ds], len(B) / n_members[ds]
            chance.append(p * q / (p + q - p * q) if p + q > 0 else 0.0)
            A, B = set(flags[(ds, alpha, agg, s1, att)][2]), set(flags[(ds, alpha, agg, s2, att)][2])
            if A | B:
                jac_nm.append(len(A & B) / len(A | B))
        counts = {}
        for sd in seeds:
            for i in flags[(ds, alpha, agg, sd, att)][0]:
                counts[i] = counts.get(i, 0) + 1
        hist = np.bincount(list(counts.values()), minlength=len(seeds) + 1)
        stab.append({"dataset": ds, "alpha": alpha, "agg": agg, "attack": name, "round": r,
                     "n_seeds": len(seeds), "jaccard@0.05": np.mean(jac) if jac else np.nan,
                     "chance_jaccard": np.mean(chance) if chance else np.nan,
                     "nonmember_jaccard@0.05": np.mean(jac_nm) if jac_nm else np.nan,
                     "flagged_in_all_seeds": int(hist[len(seeds)]),
                     "flagged_in_any_seed": int(hist[1:].sum())})
    pd.DataFrame(stab).to_csv(os.path.join(a.out, "stability.csv"), index=False)
    ident = []
    for (ds, alpha, agg, seed, att), (f5, f1, _) in flags.items():
        if not att.endswith("@100"):
            continue
        for i in f5:
            ident.append((ds, alpha, agg, seed, att, int(i), int(i in set(f1))))
    pd.DataFrame(ident, columns=["dataset", "alpha", "agg", "seed", "attack", "pool_index",
                                 "also_at_fpr_0.01"]).to_csv(
        os.path.join(a.out, "identified_samples.csv.gz"), index=False)


if __name__ == "__main__":
    main()
