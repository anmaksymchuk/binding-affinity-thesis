"""
Within-target ranking across the folds of a cross-validation run.

This is the headline metric for "does the model order ligands of the same
target", and it needs only each fold's test_predictions.csv plus the official
CASF targets from CoreSet.dat: no similarity matrices, so it is cheap to run
for all five folds. The search baseline, the Tanimoto bins and the full set of
decompositions live in tools/casf_ranking.py, which needs the GEMS matrices
and the similarity analysis of one run.

Reported per test set with official targets (CASF-2016, CASF-2013):
    Spearman and Kendall within target, averaged over targets
    median and quartiles of the per-target Spearman (the box GEMS Fig. 3c shows)
    low-level success  (the strongest ligand of the target ranked first)
    high-level success (the whole target ordered correctly)
    pairwise dpK: RMSE, sign accuracy, and the slope of predicted on true dpK
each as mean +- std over folds, and once for the ensemble of the fold models
(the mean of their predictions), which is a different model.

Usage (from the repo root):
    python tools/ranking_folds.py --coreset /path/CoreSet2016.dat \
                                  --coreset2013 /path/CoreSet2013.dat \
                                  [--prefix gnn_only_noH_amp] [--seed 42]

Writes per_fold.csv and summary.csv into outputs/cv_summary_<prefix>/ranking/.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

from casf_ranking import read_coreset, per_target, pairs, pair_stats

TARGET_METRICS = ["spearman", "kendall", "low_level_success", "high_level_success"]
# Quartiles of the per-target Spearman. GEMS Fig. 3c reports the distribution
# over the 57 clusters as a boxplot, so a comparison with it needs the median
# and the box, not the mean.
BOX_METRICS = ["spearman_median", "spearman_q1", "spearman_q3"]
PAIR_METRICS = ["rmse_dpk", "sign_accuracy", "slope_dyhat_on_dy"]


def ranking_of(g):
    """Within-target metrics of one prediction set (needs cluster, y_true, y_pred)."""
    pt = per_target(g, "y_pred")
    ok = pt.dropna(subset=["spearman"])
    pr = pairs(g, "y_pred")
    ps = pair_stats(pr)

    return {
        "n": len(g),
        "n_targets": len(ok),
        **{m: float(ok[m].mean()) for m in TARGET_METRICS},
        "spearman_median": float(ok["spearman"].median()),
        "spearman_q1": float(ok["spearman"].quantile(0.25)),
        "spearman_q3": float(ok["spearman"].quantile(0.75)),
        **{m: float(ps[m]) for m in PAIR_METRICS},
        "n_pairs": int(ps["n_pairs"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coreset", default=None, help="CASF-2016 CoreSet.dat")
    ap.add_argument("--coreset2013", default=None, help="CASF-2013 CoreSet.dat")
    ap.add_argument("--prefix", default="gnn_only_noH_amp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    official = {k: v for k, v in [("casf2016", args.coreset), ("casf2013", args.coreset2013)] if v}
    if not official:
        raise SystemExit("Give at least one of --coreset / --coreset2013 "
                         "(the official CASF targets; without them the targets would be guessed).")

    out_dir = args.out_dir or os.path.join(args.outputs, f"cv_summary_{args.prefix}", "ranking")
    os.makedirs(out_dir, exist_ok=True)

    labels = {s: read_coreset(p) for s, p in official.items()}
    for s, lab in labels.items():
        print(f"{s}: {lab.nunique()} targets from {os.path.basename(official[s])}")

    rows, preds = [], []

    for k in args.folds:
        run = os.path.join(args.outputs, f"{args.prefix}_fold{k}_seed{args.seed}")
        path = os.path.join(run, "test_predictions.csv")

        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. Check --prefix, --seed and --folds.")

        t = pd.read_csv(path, dtype={"pdb_id": str})
        t["fold"] = k
        preds.append(t)

        for s, lab in labels.items():
            g = t[t["test_set"] == s].copy()
            g["cluster"] = g["pdb_id"].map(lab)
            missing = int(g["cluster"].isna().sum())
            g = g.dropna(subset=["cluster"])
            rows.append({"fold": k, "set": s, "no_target": missing, **ranking_of(g)})

    per_fold = pd.DataFrame(rows)

    # Ensemble: the mean prediction per complex over the folds.
    ens_rows = []
    all_preds = pd.concat(preds)
    for s, lab in labels.items():
        sub = all_preds[all_preds["test_set"] == s]
        g = (sub.groupby("pdb_id")
             .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
             .reset_index())
        g["cluster"] = g["pdb_id"].map(lab)
        g = g.dropna(subset=["cluster"])
        ens_rows.append({"set": s, **ranking_of(g)})
    ensemble = pd.DataFrame(ens_rows)

    cols = TARGET_METRICS + BOX_METRICS + PAIR_METRICS
    summary = per_fold.groupby("set")[cols].agg(["mean", "std"]).reindex(list(official))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "n_targets", per_fold.groupby("set")["n_targets"].first().reindex(list(official)))

    per_fold.to_csv(os.path.join(out_dir, "per_fold.csv"), index=False)
    summary.round(4).to_csv(os.path.join(out_dir, "summary.csv"))
    ensemble.round(4).to_csv(os.path.join(out_dir, "ensemble.csv"), index=False)

    print("\nSpearman within target, per fold:")
    print(f"{'set':<12}" + "".join(f"{'f' + str(k):>9}" for k in args.folds))
    for s in official:
        sub = per_fold[per_fold["set"] == s].set_index("fold")
        print(f"{s:<12}" + "".join(f"{sub.loc[k, 'spearman']:>9.3f}" for k in args.folds))

    print(f"\nMean +- std over {len(args.folds)} folds (single models):")
    for s in official:
        r = summary.loc[s]
        print(f"\n{s} ({int(r['n_targets'])} targets with a ranking)")
        for m in cols:
            print(f"  {m:<22}{r[m + '_mean']:>8.3f} ±{r[m + '_std']:>6.3f}")

    print("\nEnsemble of the fold models (mean of predictions, a different model):")
    for r in ens_rows:
        print(f"  {r['set']:<12}" + "  ".join(f"{m} {r[m]:.3f}" for m in
                                              ["spearman", "kendall", "rmse_dpk", "slope_dyhat_on_dy"]))
        print(f"  {'':<12}per-target Spearman median {r['spearman_median']:.3f} "
              f"[{r['spearman_q1']:.3f}, {r['spearman_q3']:.3f}]  "
              f"(GEMS Fig. 3c reports this box for its own ensemble)")

    if int(per_fold["no_target"].max()):
        n = int(per_fold.groupby("set")["no_target"].first().sum())
        print(f"\nNote: {n} predicted complexes are not in the CoreSet files and were left out.")

    print(f"\nSaved {out_dir}/per_fold.csv, summary.csv and ensemble.csv")


if __name__ == "__main__":
    main()
