"""
Per-target figure for CASF, in the style of GEMS Extended Data Fig. 1:
for every CASF target (cluster of 5 complexes in CASF-2016, 3 in CASF-2013)
  row 1: distribution of |error| of the complexes of the target,
  row 2: Pearson correlation of prediction vs true pK within the target,
  row 3: Spearman correlation within the target (CASF ranking power),
for the model (left) and the search baseline top-5 (right).
Dashed lines: RMSE / Pearson on the whole set, and the mean Spearman.

Needs the outputs of tools/similarity_analysis.py (similarity_per_complex.csv)
and tools/casf_ranking.py (ranking/casf_clusters.csv; run it with --coreset /
--coreset2013 to use the official CASF targets).

Usage (from the repo root):
    python tools/plot_per_target.py [--run_dir outputs/gnn_only_val15_seed42] [--sets casf2016 casf2013]

Writes <run_dir>/similarity/ranking/per_target_<set>.png and .pdf.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from scipy import stats

PREDICTORS = [("model", "y_pred", "GNN", "#2a78d6"), ("search", "knn5_pred", "Search top-5", "#eb6834")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def per_target(g, col):
    rows = []
    for cl, sub in g.groupby("cluster"):
        ok = len(sub) >= 3 and sub[col].nunique() > 1 and sub["y_true"].nunique() > 1
        rows.append({
            "cluster": cl, "n": len(sub),
            "abs_err": (sub[col] - sub["y_true"]).abs().to_numpy(),
            "pearson": np.corrcoef(sub[col], sub["y_true"])[0, 1] if ok else np.nan,
            "spearman": stats.spearmanr(sub[col], sub["y_true"])[0] if ok else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--sets", nargs="+", default=["casf2016", "casf2013"])
    args = ap.parse_args()
    if args.run_dir is None:
        from config import Config
        args.run_dir = Config().output_dir

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = os.path.join(args.run_dir, "similarity")
    d = pd.read_csv(os.path.join(sim, "similarity_per_complex.csv"), dtype={"pdb_id": str})
    cl = pd.read_csv(os.path.join(sim, "ranking", "casf_clusters.csv"), dtype={"pdb_id": str, "cluster": str})

    for s in args.sets:
        g = d[(d["set"] == s) & d["y_pred"].notna()].merge(
            cl[cl["set"] == s][["pdb_id", "cluster"]], on="pdb_id", how="inner")
        if g.empty:
            print(f"{s}: no data, skipped")
            continue
        order = sorted(g["cluster"].unique(), key=lambda c: (float(c) if c.replace(".", "").isdigit() else np.inf, c))
        x = np.arange(1, len(order) + 1)

        fig, axes = plt.subplots(3, 2, figsize=(15, 9.5), sharey="row")
        for j, (name, col, label, color) in enumerate(PREDICTORS):
            pt = per_target(g, col).set_index("cluster").loc[order]
            e = g[col] - g["y_true"]
            rmse = np.sqrt(np.mean(e ** 2))
            r_all = np.corrcoef(g[col], g["y_true"])[0, 1]
            rho_mean = pt["spearman"].mean()

            ax = axes[0, j]
            bp = ax.boxplot(pt["abs_err"].tolist(), positions=x, widths=0.6, patch_artist=True,
                            showfliers=False, medianprops={"color": INK, "lw": 1.2},
                            whiskerprops={"color": INK2, "lw": 0.8}, capprops={"color": INK2, "lw": 0.8},
                            boxprops={"lw": 0.8, "edgecolor": INK2})
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            ax.axhline(rmse, color=INK, ls="--", lw=1)
            ax.text(len(x) + 0.6, rmse, f"RMSE {rmse:.2f}", va="bottom", ha="right", fontsize=8, color=INK,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
            ax.set_title(f"{label}: |error| per target", fontsize=11, color=INK)
            if j == 0:
                ax.set_ylabel("|prediction error|, pK")

            for i, (metric, ref, ref_lab, ylab) in enumerate([
                    ("pearson", r_all, f"Pearson on all {len(g)}: {r_all:.2f}", "Pearson within target"),
                    ("spearman", rho_mean, f"mean Spearman {rho_mean:.2f}", "Spearman within target")]):
                ax = axes[i + 1, j]
                v = pt[metric].to_numpy()
                ax.bar(x, np.nan_to_num(v), width=0.75, color=color, edgecolor="white", linewidth=0.5)
                ax.axhline(0, color=INK2, lw=0.8)
                ax.axhline(ref, color=INK, ls="--", lw=1)
                ax.text(len(x) + 0.6, ref, ref_lab, va="bottom", ha="right", fontsize=8, color=INK,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
                ax.set_ylim(-1.05, 1.1)
                ax.set_title(f"{label}: {ylab}", fontsize=11, color=INK)
                if j == 0:
                    ax.set_ylabel(ylab)

        for ax in axes.flat:
            ax.set_xlim(0.3, len(x) + 0.7)
            ticks = [t for t in x if t % 5 == 0]
            ax.set_xticks(ticks)
            ax.set_xticklabels([order[t - 1] for t in ticks], fontsize=8, color=INK2)
            ax.tick_params(axis="y", labelsize=8, colors=INK2)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(INK2)
        for ax in axes[-1]:
            ax.set_xlabel("CASF target (cluster ID)", color=INK2)
        fig.suptitle(f"{s.upper()}: per-target performance, {len(order)} targets", fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(sim, "ranking", f"per_target_{s}.{ext}"), dpi=200)
        plt.close(fig)
        print(f"Saved {os.path.join(sim, 'ranking', f'per_target_{s}.png')} (and .pdf)")


if __name__ == "__main__":
    main()
