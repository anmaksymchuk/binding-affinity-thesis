"""
Tables and plots from similarity/similarity_per_complex.csv
(the output of tools/similarity_analysis.py).

Questions it answers:
  1. Does the error depend on LIGAND similarity to train (max Tanimoto)?
     (max Tanimoto + TM is dominated by TM ~ 1, so the ligand part is shown alone)
  2. Does the error depend on how different the label of the nearest training
     neighbour is (|pK - pK of the nearest neighbour|)? CleanSplit removes
     similar training complexes only if their pK differs by < 1, so similar
     neighbours of CASF complexes tend to have a different affinity.
  3. Does the model compress predictions towards the mean
     (slope of y_pred on y_true, bias per affinity range)?
  4. How often does the model beat the search baseline (top-5 neighbours)?

RMSE per bin comes with a 95% bootstrap interval; bins with few complexes
are wide, read them with care.

Usage (from the repo root):
    python tools/similarity_report.py [--run_dir outputs/gnn_only_val15_seed42]

Writes CSV tables and PNG plots to <run_dir>/similarity/report/.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

# Fixed colour per set (categorical palette, fixed order) and marker as a
# second cue, so sets are not distinguished by colour alone.
SET_STYLE = {
    "val": ("#2a78d6", "o"),
    "casf2016": ("#eb6834", "s"),
    "casf2016_indep": ("#1baf7a", "^"),
    "casf2013": ("#eda100", "D"),
    "casf2013_indep": ("#e87ba4", "v"),
    "train (sample, self excluded)": ("#6250d6", "x"),
}
MAIN_SETS = ["val", "casf2016", "casf2016_indep"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

TAN_BINS = [0.0, 0.4, 0.6, 0.8, 0.9, 1.0001]
TAN_LABELS = ["<0.4", "0.4-0.6", "0.6-0.8", "0.8-0.9", ">=0.9"]
DPK_BINS = [0.0, 0.5, 1.0, 2.0, np.inf]
DPK_LABELS = ["<0.5", "0.5-1", "1-2", ">=2"]
PK_BINS = [-np.inf, 4.0, 6.0, 8.0, np.inf]
PK_LABELS = ["<4", "4-6", "6-8", ">=8"]


def rmse(e):
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def boot_ci(e, n_boot=1000, seed=0):
    e = np.asarray(e, dtype=float)
    if len(e) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = [rmse(rng.choice(e, len(e), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def binned_table(df, col, bins, labels):
    rows = []
    for name, g in df.groupby("set", sort=False):
        b = pd.cut(g[col], bins=bins, labels=labels, right=False)
        for lab in labels:
            s = g[b == lab]
            err = (s["y_pred"] - s["y_true"]).to_numpy()
            kerr = (s["knn5_pred"] - s["y_true"]).to_numpy()
            lo, hi = boot_ci(err)
            rows.append({
                "set": name, "bin": lab, "n": len(s),
                "model_rmse": rmse(err), "ci_low": lo, "ci_high": hi,
                "knn5_rmse": rmse(kerr),
                "model_bias": float(err.mean()) if len(s) else np.nan,
            })
    return pd.DataFrame(rows)


def calibration_table(df):
    rows = []
    for name, g in df.groupby("set", sort=False):
        slope, intercept = np.polyfit(g["y_true"], g["y_pred"], 1)
        err = g["y_pred"] - g["y_true"]
        row = {
            "set": name, "n": len(g),
            "slope_pred_on_true": slope, "intercept": intercept,
            "std_pred / std_true": g["y_pred"].std() / g["y_true"].std(),
            "model_beats_knn5_frac": float((g["abs_err"] < g["knn5_abs_err"]).mean()),
            "r_model_knn5": float(np.corrcoef(g["y_pred"], g["knn5_pred"])[0, 1]),
        }
        b = pd.cut(g["y_true"], bins=PK_BINS, labels=PK_LABELS, right=False)
        for lab in PK_LABELS:
            e = err[b == lab]
            row[f"n pK {lab}"] = int(len(e))
            row[f"bias pK {lab}"] = float(e.mean()) if len(e) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_binned(table, title, xlabel, path, plt):
    sets = [s for s in MAIN_SETS if s in set(table["set"])]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.8), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], sets):
        t = table[(table["set"] == name) & (table["n"] > 0)]
        x = np.arange(len(t))
        color, marker = SET_STYLE[name]
        yerr = np.vstack([t["model_rmse"] - t["ci_low"], t["ci_high"] - t["model_rmse"]])
        ax.errorbar(x, t["model_rmse"], yerr=np.nan_to_num(yerr), color=color, marker=marker,
                    markersize=7, linewidth=2, capsize=3, label="model")
        ax.plot(x, t["knn5_rmse"], color=INK2, marker="o", markersize=6, linewidth=1.5,
                linestyle="--", label="search top-5")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}\nn={n}" for b, n in zip(t["bin"], t["n"])], fontsize=8)
        ax.set_title(name, color=INK, fontsize=11)
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
        style_axes(ax)
    axes[0][0].set_ylabel("RMSE, pK", color=INK2)
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None)
    args = ap.parse_args()
    if args.run_dir is None:
        from config import Config
        args.run_dir = Config().output_dir

    src = os.path.join(args.run_dir, "similarity", "similarity_per_complex.csv")
    out = os.path.join(args.run_dir, "similarity", "report")
    os.makedirs(out, exist_ok=True)

    df_all = pd.read_csv(src, dtype={"pdb_id": str})
    df_all["dpk_nn"] = (df_all["y_true"] - df_all["nn_pk"]).abs()
    df = df_all[df_all["y_pred"].notna()].copy()
    if df.empty:
        sys.exit("No model predictions in the CSV: run tools/evaluate.py and similarity_analysis.py first.")

    t_tan = binned_table(df, "max_tanimoto", TAN_BINS, TAN_LABELS)
    t_dpk = binned_table(df, "dpk_nn", DPK_BINS, DPK_LABELS)
    t_cal = calibration_table(df)
    t_tan.to_csv(os.path.join(out, "rmse_by_ligand_similarity.csv"), index=False)
    t_dpk.to_csv(os.path.join(out, "rmse_by_dpk_nearest_neighbour.csv"), index=False)
    t_cal.to_csv(os.path.join(out, "calibration_and_vs_knn.csv"), index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    show = ["set", "bin", "n", "model_rmse", "ci_low", "ci_high", "knn5_rmse", "model_bias"]
    print("=== 1. RMSE by ligand similarity to train (max Tanimoto) ===")
    print(t_tan[t_tan["n"] > 0][show].round(3).to_string(index=False))
    print("\n=== 2. RMSE by |pK - pK of nearest training neighbour| (neighbour by Tanimoto + TM) ===")
    print(t_dpk[t_dpk["n"] > 0][show].round(3).to_string(index=False))
    print("\n=== 3-4. Compression towards the mean, bias by affinity range, model vs search ===")
    print(t_cal.round(3).T.to_string(header=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("matplotlib not installed, plots skipped")
        return

    plot_binned(t_tan, "Error by ligand similarity to the training set", "max Tanimoto to train",
                os.path.join(out, "rmse_by_ligand_similarity.png"), plt)
    plot_binned(t_dpk, "Error by label difference to the nearest training complex",
                "|pK - pK of nearest neighbour|", os.path.join(out, "rmse_by_dpk_nearest_neighbour.png"), plt)

    # Predicted vs true, coloured by ligand similarity (one-hue sequential ramp)
    blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    sets = [s for s in MAIN_SETS if s in set(df["set"])]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.4 * len(sets), 4.2), squeeze=False)
    lim = (min(df["y_true"].min(), df["y_pred"].min()) - 0.5, max(df["y_true"].max(), df["y_pred"].max()) + 0.5)
    for ax, name in zip(axes[0], sets):
        g = df[df["set"] == name].sort_values("max_tanimoto")
        sc = ax.scatter(g["y_true"], g["y_pred"], c=g["max_tanimoto"], cmap=blues, vmin=0.2, vmax=1.0,
                        s=14, edgecolors="none", alpha=0.85)
        ax.plot(lim, lim, color=INK2, linewidth=1, linestyle="--")
        k, b = np.polyfit(g["y_true"], g["y_pred"], 1)
        xs = np.array(lim)
        ax.plot(xs, k * xs + b, color=INK, linewidth=2)
        ax.text(0.03, 0.95, f"slope {k:.2f}\nRMSE {rmse(g['y_pred'] - g['y_true']):.2f}", transform=ax.transAxes,
                va="top", fontsize=9, color=INK)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(name, color=INK, fontsize=11)
        ax.set_xlabel("true pK", color=INK2)
        style_axes(ax)
    axes[0][0].set_ylabel("predicted pK", color=INK2)
    cb = fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8)
    cb.set_label("max Tanimoto to train", color=INK2)
    fig.suptitle("Predicted vs true (dashed: ideal, solid: fitted line)", color=INK, fontsize=12)
    fig.savefig(os.path.join(out, "pred_vs_true.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Bias by affinity range
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for name, g in df.groupby("set", sort=False):
        color, marker = SET_STYLE.get(name, (INK2, "o"))
        b = pd.cut(g["y_true"], bins=PK_BINS, labels=PK_LABELS, right=False)
        bias = [(g["y_pred"] - g["y_true"])[b == lab].mean() for lab in PK_LABELS]
        ax.plot(PK_LABELS, bias, color=color, marker=marker, markersize=7, linewidth=2, label=name)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xlabel("true pK range", color=INK2)
    ax.set_ylabel("mean (predicted - true), pK", color=INK2)
    ax.set_title("Bias by affinity range (above 0: overestimated)", color=INK, fontsize=11)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bias_by_affinity.png"), dpi=150)
    plt.close(fig)

    # Ligand similarity distributions, including the train reference sample
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bins = np.linspace(0, 1, 41)
    for name, g in df_all.groupby("set", sort=False):
        if name in ("casf2013", "casf2013_indep"):
            continue
        color, _ = SET_STYLE.get(name, (INK2, "o"))
        ax.hist(g["max_tanimoto"], bins=bins, density=True, histtype="step", linewidth=2,
                color=color, label=f"{name} (n={len(g)})")
    ax.set_xlabel("max Tanimoto to train", color=INK2)
    ax.set_ylabel("density", color=INK2)
    ax.set_title("Ligand similarity to the closest training complex", color=INK, fontsize=11)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ligand_similarity_hist.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved tables and plots to {out}")


if __name__ == "__main__":
    main()
