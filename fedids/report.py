"""Tables (LaTeX) and figures for the federated audit from the CSVs written by analyze.py."""
import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})
AGGS = ["fedavg", "fedprox", "scaffold", "fedadam"]
AGG_NAME = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fedadam": "FedAdam"}
ATTACKS = ["loss", "refgap", "gradnorm", "gradalign", "trajectory"]
ATT_NAME = {"loss": "Loss", "refgap": "Ref. gap", "gradnorm": "Grad. norm", "gradalign": "Grad. align",
            "trajectory": "Trajectory"}
DS_NAME = {"cicids2017": "CIC-IDS2017", "ciciot2023": "CICIoT2023"}
PALETTE = ["#1b4d5c", "#4f9d8f", "#8c4a6b", "#d9a441"]


def pm(s, d=3):
    return f"{s.mean():.{d}f} $\\pm$ {s.std(ddof=1) if len(s) > 1 else 0:.{d}f}"


def final(R):
    return R[(R["round"] == 100)]


def main_table(R, U, out):
    """One row per dataset and aggregator, mean over the three concentrations and the seeds."""
    F = final(R)
    lines = []
    for ds in [d for d in DS_NAME if d in F.dataset.unique()]:
        lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{DS_NAME[ds]}}}}} \\\\")
        for agg in AGGS:
            c = F[(F.dataset == ds) & (F["agg"] == agg)]
            if c.empty:
                continue
            f1 = c[c.attack == "loss"].test_macro_f1
            auc = [c[c.attack == a].auc.mean() for a in ATTACKS]
            best = c.groupby("attack")["tpr@0.01"].mean().max()
            u = U[(U.dataset == ds) & (U["agg"] == agg)]
            lines.append(f"{AGG_NAME[agg]} & {f1.mean():.3f} & "
                         + " & ".join(f"{v:.4f}" for v in auc)
                         + f" & {100 * best:.2f} & {100 * u.cal_tpr.mean():.2f} \\\\")
        lines.append("\\addlinespace[2pt]")
    open(os.path.join(out, "tab_fed_main.tex"), "w").write("\n".join(lines[:-1]) + "\n")


def stability_table(S, out):
    F = S[S["round"] == 100]
    dss = [d for d in DS_NAME if d in F.dataset.unique()]
    lines = []
    for att in ATTACKS:
        cells = []
        for ds in dss:
            g = F[(F.dataset == ds) & (F.attack == att)]
            cells += [f"{g['jaccard@0.05'].mean():.3f}", f"{g['nonmember_jaccard@0.05'].mean():.3f}",
                      f"{g['chance_jaccard'].mean():.3f}"]
        lines.append(f"{ATT_NAME[att]} & " + " & ".join(cells) + " \\\\")
    open(os.path.join(out, "tab_fed_stability.tex"), "w").write("\n".join(lines) + "\n")


def heatmap(R, out):
    F = final(R)
    dss = [d for d in DS_NAME if d in F.dataset.unique()]
    fig, axes = plt.subplots(len(dss), 1, figsize=(6.8, 2.2 * len(dss)), squeeze=False)
    for ax, ds in zip(axes[:, 0], dss):
        cols = [(a, g) for a in sorted(F.alpha.unique()) for g in AGGS]
        M = np.array([[F[(F.dataset == ds) & (F.alpha == a) & (F["agg"] == g) & (F.attack == t)].auc.mean()
                       for a, g in cols] for t in ATTACKS])
        im = ax.imshow(M, cmap="BrBG", vmin=0.46, vmax=0.54, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.3f}"[1:], ha="center", va="center", fontsize=6)
        ax.set_yticks(range(len(ATTACKS)), [ATT_NAME[t] for t in ATTACKS])
        ax.set_xticks(range(len(cols)), [AGG_NAME[g][:4] if g != "scaffold" else "SCAF" for _, g in cols],
                      rotation=90, fontsize=6)
        for k, a in enumerate(sorted(F.alpha.unique())):
            ax.text(4 * k + 1.5, -0.6, f"Dirichlet {a:g}", ha="center", va="bottom", fontsize=6.5)
            if k:
                ax.axvline(4 * k - 0.5, color="white", lw=1.5)
        ax.set_title(DS_NAME[ds], pad=12)
    fig.tight_layout()
    fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.8, label="AUC (round 100)")
    fig.savefig(os.path.join(out, "fig_fed_auc.pdf"), bbox_inches="tight")
    plt.close(fig)


def rounds_fig(R, out):
    dss = [d for d in DS_NAME if d in R.dataset.unique()]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.4 * len(dss), 2.0), squeeze=False, sharey=True)
    for ax, ds in zip(axes[0], dss):
        for k, agg in enumerate(AGGS):
            for att, ls in [("loss", "-"), ("gradalign", "--")]:
                c = R[(R.dataset == ds) & (R["agg"] == agg) & (R.attack == att)]
                g = c.groupby("round")["tpr@0.01"].mean() * 100
                ax.plot(g.index, g.values, ls, color=PALETTE[k], marker="o", ms=2.5,
                        label=f"{AGG_NAME[agg]}, {ATT_NAME[att]}")
        ax.axhline(1.0, color="grey", lw=0.6, ls=":")
        ax.set_xticks([10, 50, 100])
        ax.set_xlabel("Audited round")
        ax.set_title(DS_NAME[ds])
    axes[0][0].set_ylabel("TPR at 1% FPR (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.savefig(os.path.join(out, "fig_fed_rounds.pdf"), bbox_inches="tight")
    plt.close(fig)


def class_client_fig(C, K, classes, out):
    dss = [d for d in DS_NAME if d in C.dataset.unique()]
    fig, axes = plt.subplots(2, len(dss), figsize=(3.4 * len(dss), 4.0), squeeze=False)
    for j, ds in enumerate(dss):
        c = C[(C.dataset == ds) & (C["round"] == 100) & (C.attack == "loss")]
        g = c.groupby(["class", "alpha"])[["tpr@0.05", "fpr@0.05"]].mean().unstack() * 100
        alphas = sorted(c.alpha.unique())
        ax = axes[0][j]
        w = 0.8 / len(alphas)
        x = np.arange(len(g))
        for k, a in enumerate(alphas):
            off = (k - (len(alphas) - 1) / 2) * w
            ax.bar(x + off, g[("tpr@0.05", a)].values, w, color=PALETTE[k], label=f"TPR, Dirichlet {a:g}")
            ax.scatter(x + off, g[("fpr@0.05", a)].values, marker="_", s=40, color="black", lw=1.2,
                       label="FPR of the same threshold" if k == 0 else None, zorder=3)
        ax.set_xticks(x, [classes[ds][i] for i in g.index], rotation=35, ha="right", fontsize=6)
        ax.set_ylabel("Records flagged (%)")
        ax.set_title(f"{DS_NAME[ds]}: per class, Loss, 5% global FPR")
        ax.legend(fontsize=5.5, frameon=False)
        k_ = K[(K.dataset == ds) & (K["round"] == 100) & (K.attack == "loss")]
        g2 = k_.assign(ex=100 * (k_["tpr@0.05"] - k_["class_matched_fpr@0.05"])).groupby(
            ["alpha", "client", "agg"]).agg(n=("n_train", "first"), ex=("ex", "mean")).reset_index()
        ax = axes[1][j]
        for k, agg in enumerate(AGGS):
            h = g2[g2["agg"] == agg]
            ax.scatter(h.n, h.ex, s=6, color=PALETTE[k], label=AGG_NAME[agg], alpha=0.8)
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("Training records held by the client")
        ax.set_ylabel("Excess exposure (pp)")
        ax.set_title(f"{DS_NAME[ds]}: per client, all partitions")
        ax.legend(fontsize=5.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_fed_class_client.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.expanduser("~/fedids_work"))
    ap.add_argument("--res", default="results")
    a = ap.parse_args()
    R = pd.read_csv(os.path.join(a.res, "runs.csv"))
    U = pd.read_csv(os.path.join(a.res, "union.csv"))
    C = pd.read_csv(os.path.join(a.res, "per_class.csv"))
    K = pd.read_csv(os.path.join(a.res, "per_client.csv"))
    classes = {ds: json.load(open(os.path.join(a.work, "data", f"{ds}.json")))["classes"]
               for ds in R.dataset.unique()}
    main_table(R, U, a.res)
    stability_table(pd.read_csv(os.path.join(a.res, "stability.csv")), a.res)
    heatmap(R, a.res)
    rounds_fig(R, a.res)
    class_client_fig(C, K, classes, a.res)


if __name__ == "__main__":
    main()
