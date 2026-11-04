"""
Control checks on similarity/similarity_per_complex.csv (output of
tools/similarity_analysis.py). No retraining needed.

They test, more carefully than tools/similarity_report.py, the hypothesis
that the label-conditioned CleanSplit filter (a similar training complex is
removed only if |dpK| < 1) leaves CASF complexes with similar training
neighbours of DIFFERENT affinity, and how much of the val -> test gap this
explains. Note: |error| and |pK - pK of nearest neighbour| both use the true
pK, so their correlation is partly arithmetic; checks 1 and 2 below avoid or
quantify that.

  1. Enrichment, without model predictions: among complexes that have a
     training complex passing the CleanSplit structural criterion
     (n_close > 0), how often is the nearest neighbour's pK within 1 unit?
     Compared with complexes without such a neighbour, per set.
  2. Reweighting: val MSE per dpK bin, weighted by each test set's dpK mix.
     The share of the val -> test MSE gap that the dpK mix alone explains.
  3. dpK effect within fixed true-pK ranges (is it just affinity range?).
  4. Paired comparison with the search baseline: mean of
     e_search^2 - e_model^2 per complex with a bootstrap 95% CI
     (resampling complexes, not target clusters).
  5. Bias by affinity range compared with a constant predictor
     (training-set mean pK, from y_stats.pt if present).
  6. Who is the nearest neighbour: same protein (TM) / similar ligand (Tanimoto).
  7. Linear recalibration fitted on val (y_true ~ y_pred), applied to test.

Usage (from the repo root):
    python tools/similarity_checks.py [--run_dir outputs/gnn_only_val15_seed42]

Prints the tables and writes them as CSV to <run_dir>/similarity/checks/.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

DPK_BINS = [0.0, 0.5, 1.0, 2.0, np.inf]
DPK_LABELS = ["<0.5", "0.5-1", "1-2", ">=2"]
TEST_SETS = ["casf2016", "casf2016_indep", "casf2013", "casf2013_indep"]


def rmse(e):
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.run_dir is None:
        from config import Config
        args.run_dir = Config().output_dir

    src = os.path.join(args.run_dir, "similarity", "similarity_per_complex.csv")
    out = os.path.join(args.run_dir, "similarity", "checks")
    os.makedirs(out, exist_ok=True)

    d = pd.read_csv(src, dtype={"pdb_id": str})
    d = d[d["y_pred"].notna()].copy()
    d["dpk"] = (d["y_true"] - d["nn_pk"]).abs()
    d["e"] = d["y_pred"] - d["y_true"]
    d["e_knn"] = d["knn5_pred"] - d["y_true"]
    d["dpk_bin"] = pd.cut(d["dpk"], DPK_BINS, labels=DPK_LABELS, right=False)
    sets = list(dict.fromkeys(d["set"]))
    val = d[d["set"] == "val"]
    if val.empty:
        sys.exit("No validation predictions: run tools/evaluate.py, then similarity_analysis.py.")

    y_stats_path = os.path.join(args.run_dir, "y_stats.pt")
    if os.path.exists(y_stats_path):
        import torch
        train_mean = float(torch.load(y_stats_path)["mean"])
    else:
        train_mean = float(val["y_true"].mean())
        print(f"Note: y_stats.pt not found, constant predictor uses the val mean {train_mean:.3f}")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    # 1. Enrichment of conflicting neighbours (no model predictions involved)
    rows = []
    for s in sets:
        g = d[d["set"] == s]
        for has_close, sub in ((True, g[g["n_close"] > 0]), (False, g[g["n_close"] == 0])):
            rows.append({
                "set": s, "has_close_train_complex": has_close, "n": len(sub),
                "share_nn_dpk_lt_1": (sub["dpk"] < 1).mean() if len(sub) else np.nan,
                "median_nn_dpk": sub["dpk"].median() if len(sub) else np.nan,
            })
    t1 = pd.DataFrame(rows)
    print("=== 1. Nearest-neighbour label agreement, with vs without a CleanSplit-close training complex ===")
    print(t1.round(3).to_string(index=False))

    # 2. Reweighting val error by the dpK mix of each test set
    val_mse = val.groupby("dpk_bin", observed=False)["e"].apply(lambda e: np.mean(e ** 2)).reindex(DPK_LABELS)
    mse_val = float(np.mean(val["e"] ** 2))
    rows = []
    for s in sets:
        g = d[d["set"] == s]
        w = g["dpk_bin"].value_counts(normalize=True).reindex(DPK_LABELS).fillna(0)
        mse_exp = float((w * val_mse).sum())
        mse_act = float(np.mean(g["e"] ** 2))
        gap = mse_act - mse_val
        rows.append({
            "set": s, "n": len(g), **{f"share dpK {b}": w[b] for b in DPK_LABELS},
            "rmse_actual": np.sqrt(mse_act), "rmse_expected_from_val_by_dpk_mix": np.sqrt(mse_exp),
            "share_of_mse_gap_explained": (mse_exp - mse_val) / gap if s != "val" and gap > 0 else np.nan,
        })
    t2 = pd.DataFrame(rows)
    print("\n=== 2. Val error reweighted to each set's dpK mix ===")
    print(t2.round(3).to_string(index=False))

    # 3. dpK effect within fixed true-pK ranges
    rows = []
    for s in sets:
        g = d[d["set"] == s].assign(pk_range=lambda x: pd.cut(
            x["y_true"], [-np.inf, 5, 7, np.inf], labels=["<5", "5-7", ">=7"], right=False))
        for (pk, b), sub in g.groupby(["pk_range", "dpk_bin"], observed=True):
            rows.append({"set": s, "pk_range": pk, "dpk_bin": b, "n": len(sub), "rmse": rmse(sub["e"])})
    t3 = pd.DataFrame(rows)
    print("\n=== 3. RMSE by dpK bin within true-pK ranges ===")
    for s in ["val", "casf2016"]:
        if s in sets:
            piv = t3[t3["set"] == s].pivot(index="pk_range", columns="dpk_bin", values=["n", "rmse"])
            piv = piv.reindex(index=["<5", "5-7", ">=7"]).reindex(columns=DPK_LABELS, level=1)
            print(s)
            print(piv.round(2).to_string())

    # 4. Paired comparison with the search baseline
    rng = np.random.default_rng(args.seed)
    rows = []
    for s in sets:
        g = d[d["set"] == s]
        x = (g["e_knn"] ** 2 - g["e"] ** 2).to_numpy()
        boots = [rng.choice(x, len(x), replace=True).mean() for _ in range(args.n_boot)]
        rows.append({
            "set": s, "n": len(x), "mean_esearch2_minus_emodel2": x.mean(),
            "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
            "share_model_better": (x > 0).mean(),
        })
    t4 = pd.DataFrame(rows)
    print("\n=== 4. Paired: mean(e_search^2 - e_model^2), bootstrap over complexes ===")
    print(t4.round(3).to_string(index=False))

    # 5. Bias vs constant predictor
    rows = []
    for s in sets:
        g = d[d["set"] == s]
        b = pd.cut(g["y_true"], [-np.inf, 4, 6, 8, np.inf], labels=["<4", "4-6", "6-8", ">=8"], right=False)
        for lab, sub in g.groupby(b, observed=True):
            mb, cb = sub["e"].mean(), (train_mean - sub["y_true"]).mean()
            rows.append({"set": s, "pk_range": lab, "n": len(sub), "model_bias": mb,
                         "constant_bias": cb, "model_bias / constant_bias": mb / cb if cb else np.nan})
        rows.append({"set": s, "pk_range": "all", "n": len(g), "model_bias": g["e"].mean(),
                     "constant_bias": (train_mean - g["y_true"]).mean(),
                     "model_rmse": rmse(g["e"]), "constant_rmse": rmse(train_mean - g["y_true"])})
    t5 = pd.DataFrame(rows)
    print(f"\n=== 5. Bias by affinity range vs constant predictor (train mean {train_mean:.3f}) ===")
    print(t5.round(3).to_string(index=False))

    # 6. Who is the nearest neighbour
    rows = []
    for s in sets:
        g = d[d["set"] == s]
        rows.append({
            "set": s, "n": len(g),
            "median_nn_tm": g["nn_tm"].median(), "share_nn_tm_gt_0.9": (g["nn_tm"] > 0.9).mean(),
            "median_nn_tanimoto": g["nn_tanimoto"].median(),
            "share_nn_tanimoto_lt_0.5": (g["nn_tanimoto"] < 0.5).mean(),
            "share_nn_tanimoto_ge_0.9": (g["nn_tanimoto"] >= 0.9).mean(),
            "share_nn_has_max_tanimoto": np.isclose(g["nn_tanimoto"], g["max_tanimoto"]).mean(),
        })
    t6 = pd.DataFrame(rows)
    print("\n=== 6. Nearest neighbour (by Tanimoto + TM): protein vs ligand similarity ===")
    print(t6.round(3).to_string(index=False))

    # 7. Linear recalibration fitted on val
    a, b0 = np.polyfit(val["y_pred"], val["y_true"], 1)
    rows = []
    for s in sets:
        g = d[d["set"] == s]
        rows.append({"set": s, "rmse": rmse(g["e"]), "rmse_recalibrated": rmse(a * g["y_pred"] + b0 - g["y_true"]),
                     "slope_pred_on_true": np.polyfit(g["y_true"], g["y_pred"], 1)[0],
                     "pearson": np.corrcoef(g["y_pred"], g["y_true"])[0, 1],
                     "std_true": g["y_true"].std(), "std_pred": g["y_pred"].std()})
    t7 = pd.DataFrame(rows)
    print(f"\n=== 7. Linear recalibration fitted on val: y = {a:.3f} * pred + {b0:.3f} ===")
    print(t7.round(3).to_string(index=False))

    for name, t in [("1_neighbour_label_agreement", t1), ("2_reweighting_by_dpk", t2),
                    ("3_dpk_within_pk_ranges", t3), ("4_paired_vs_search", t4),
                    ("5_bias_vs_constant", t5), ("6_nearest_neighbour_type", t6),
                    ("7_recalibration", t7)]:
        t.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    print(f"\nSaved tables to {out}")


if __name__ == "__main__":
    main()
