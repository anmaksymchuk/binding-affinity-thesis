"""
Paired comparison of three models on exactly the same complexes.

    full    the GNN on complex graphs      outputs/<full>_fold{k}_seed{s}/test_predictions.csv
    ligand  the same GNN on ligand graphs  outputs/<ligand>_fold{k}_seed{s}/test_predictions.csv
    fp      Morgan fingerprints + boosting outputs/<full>_fold{k}_seed{s}/ligand_fp/predictions.csv

The three are trained on the same folds of the same split, so the comparison is
paired per complex. Everything is computed on the intersection of the pdb_ids
that all three predict in every fold: the fingerprint baseline silently drops
ligands RDKit cannot read, and comparing two sets that differ by a few
complexes is not a comparison of the models.

How the folds are combined (the rule we agreed on):
    the squared error of a complex is averaged over the folds, never the
    predictions. Averaging predictions would build an ensemble, which is a
    different model; averaging e^2 estimates how a single model does.
    Ranking is combined the same way: the Spearman of a target is computed
    inside each fold first and then averaged over folds.

Uncertainty: a paired bootstrap over CASF targets (the protein clusters of
CoreSet.dat), not over complexes, because complexes of one target are not
independent. The same resampled targets are used for every model, so the
bootstrap measures the difference and not the two noisy numbers separately.
The independent subsets have no targets of their own, so the targets of the
parent set are used for them; within-target ranking is not reported there
because their groups are incomplete.

Usage (from the repo root):
    python tools/compare_models.py --coreset /path/CoreSet2016.dat \
                                   --coreset2013 /path/CoreSet2013.dat \
                                   [--full gnn_only_noH_amp] \
                                   [--ligand gnn_only_noH_amp_ligandonly] \
                                   [--folds 0 1 2 3 4] [--seed 42] [--n_boot 2000]

Writes per_fold.csv, summary.csv and paired.csv into
outputs/cv_summary_<full>/compare/.
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
from trainer import Trainer

MODELS = ["full", "ligand", "fp"]
LABELS = {"full": "GNN complex", "ligand": "GNN ligand only", "fp": "Morgan + boosting"}
# Which CoreSet file gives the targets of each test set. The independent
# subsets are subsets of their parent set, so they inherit its targets.
PARENT = {"casf2016": "casf2016", "casf2016_indep": "casf2016",
          "casf2013": "casf2013", "casf2013_indep": "casf2013"}


def metrics_of(y_true, y_pred):
    e = y_pred - y_true
    return {
        "n": len(y_true),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "ci": float(Trainer.concordance_index(y_true, y_pred)),
        "pearson": float(np.corrcoef(y_pred, y_true)[0, 1]),
    }


def read_predictions(path, set_col):
    t = pd.read_csv(path, dtype={"pdb_id": str})
    if set_col not in t.columns:
        raise SystemExit(f"{path}: no column '{set_col}' (found {list(t.columns)})")
    return t.rename(columns={set_col: "set"})[["set", "pdb_id", "y_true", "y_pred"]]


def load(args):
    """One long table: model, fold, set, pdb_id, y_true, y_pred."""
    frames = []

    for k in args.folds:
        full_dir = os.path.join(args.outputs, f"{args.full}_fold{k}_seed{args.seed}")
        ligand_dir = os.path.join(args.outputs, f"{args.ligand}_fold{k}_seed{args.seed}")

        sources = {
            "full": (os.path.join(full_dir, "test_predictions.csv"), "test_set"),
            "ligand": (os.path.join(ligand_dir, "test_predictions.csv"), "test_set"),
            "fp": (os.path.join(full_dir, "ligand_fp", "predictions.csv"), "set"),
        }

        for model, (path, set_col) in sources.items():
            if not os.path.exists(path):
                raise SystemExit(
                    f"{path} not found.\n"
                    f"  full   runs come from main.py with graph_input='complex'\n"
                    f"  ligand runs come from main.py with graph_input='ligand'\n"
                    f"  fp     comes from tools/ligand_fp_baseline.py <full run dir>")
            t = read_predictions(path, set_col)
            t["model"] = model
            t["fold"] = k
            frames.append(t)

    d = pd.concat(frames, ignore_index=True)
    # The fingerprint file also holds train and val; keep the test sets only.
    return d[d["set"].isin(PARENT)].copy()


def common_ids(d, n_expected):
    """IDs predicted by every model in every fold, per test set, and what was dropped."""
    keep, dropped = [], []

    for name, g in d.groupby("set"):
        counts = g.groupby("pdb_id")["model"].count()
        ids = set(counts[counts == n_expected].index)
        missing = sorted(set(counts.index) - ids)
        keep.append(g[g["pdb_id"].isin(ids)])
        dropped.append({"set": name, "n_common": len(ids), "n_dropped": len(missing),
                        "dropped_ids": " ".join(missing[:10])})

    return pd.concat(keep, ignore_index=True), pd.DataFrame(dropped)


def check_labels(d):
    """The same complex must carry the same label everywhere, or these are not the same data."""
    spread = d.groupby(["set", "pdb_id"])["y_true"].agg(lambda s: s.max() - s.min())
    worst = float(spread.max()) if len(spread) else 0.0

    if worst > 1e-4:
        bad = spread.idxmax()
        raise SystemExit(f"The label of {bad[1]} in {bad[0]} differs between models or folds "
                         f"(gap {worst:.3f} pK). The runs are not on the same data.")


def per_complex_e2(d):
    """Mean over folds of the squared error of each complex, per model."""
    d = d.assign(e2=(d["y_pred"] - d["y_true"]) ** 2)
    return (d.groupby(["set", "model", "pdb_id"])["e2"].mean()
            .unstack("model").reset_index())


def per_target_spearman(d, labels):
    """Mean over folds of the within-target Spearman, per model and target."""
    rows = []

    for (name, model, fold), g in d.groupby(["set", "model", "fold"]):
        g = g.copy()
        g["cluster"] = g["pdb_id"].map(labels[PARENT[name]])
        g = g.dropna(subset=["cluster"])
        pt = per_target(g, "y_pred").dropna(subset=["spearman"])
        for _, r in pt.iterrows():
            rows.append({"set": name, "model": model, "fold": fold,
                         "cluster": r["cluster"], "spearman": r["spearman"]})

    if not rows:
        return pd.DataFrame(columns=["set", "cluster"] + MODELS)

    s = pd.DataFrame(rows)

    return (s.groupby(["set", "model", "cluster"])["spearman"].mean()
            .unstack("model").reset_index())


def boot_diff_weighted(values, groups, n_boot, rng):
    """
    Paired bootstrap of a mean over complexes, resampling targets.

    The complexes of a target move together, so the resampled mean is
    sum(target sums) / sum(target counts) and never a mean of means.
    """
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    agg = df.groupby("g")["v"].agg(["sum", "count"])
    sums, counts = agg["sum"].to_numpy(), agg["count"].to_numpy()
    n = len(sums)

    if n == 0:
        return np.nan, np.nan, np.nan

    idx = rng.integers(0, n, size=(n_boot, n))
    draws = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)

    return float(sums.sum() / counts.sum()), float(np.percentile(draws, 2.5)), \
        float(np.percentile(draws, 97.5))


def boot_diff_targets(values, n_boot, rng):
    """Paired bootstrap of a mean over targets (one value per target)."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = len(v)

    if n == 0:
        return np.nan, np.nan, np.nan

    draws = v[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)

    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coreset", default=None, help="CASF-2016 CoreSet.dat")
    ap.add_argument("--coreset2013", default=None, help="CASF-2013 CoreSet.dat")
    ap.add_argument("--full", default="gnn_only_noH_amp")
    ap.add_argument("--ligand", default="gnn_only_noH_amp_ligandonly")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    official = {k: v for k, v in [("casf2016", args.coreset),
                                  ("casf2013", args.coreset2013)] if v}
    if not official:
        raise SystemExit("Give at least one of --coreset / --coreset2013: the bootstrap "
                         "resamples CASF targets, which only those files define.")

    labels = {s: read_coreset(p) for s, p in official.items()}
    out_dir = args.out_dir or os.path.join(args.outputs, f"cv_summary_{args.full}", "compare")
    os.makedirs(out_dir, exist_ok=True)

    d = load(args)
    check_labels(d)
    d, dropped = common_ids(d, n_expected=len(MODELS) * len(args.folds))
    sets = [s for s in PARENT if s in set(d["set"]) and PARENT[s] in labels]
    skipped = sorted(set(d["set"]) - set(sets))
    # Without the CoreSet file of its parent a set has no targets, so it cannot
    # be bootstrapped; drop it here rather than half-way through the analysis.
    d = d[d["set"].isin(sets)].copy()

    print(f"Folds:  {args.folds}, seed {args.seed}")
    print(f"Models: " + ", ".join(f"{m} ({LABELS[m]})" for m in MODELS))
    print("\nComplexes used (predicted by all three models in every fold):")
    for _, r in dropped[dropped["set"].isin(sets)].iterrows():
        note = f"  [{r['n_dropped']} dropped: {r['dropped_ids']}]" if r["n_dropped"] else ""
        print(f"  {r['set']:<16}{r['n_common']:>5}{note}")

    if skipped:
        print(f"\nSkipped (no CoreSet file for their targets): {', '.join(skipped)}")

    # ---------------- per fold and mean +- std ----------------
    rows = []
    for (name, model, fold), g in d.groupby(["set", "model", "fold"]):
        r = {"set": name, "model": model, "fold": fold,
             **metrics_of(g["y_true"].to_numpy(), g["y_pred"].to_numpy())}

        if PARENT[name] == name:     # ranking only where the groups are complete
            g = g.copy()
            g["cluster"] = g["pdb_id"].map(labels[PARENT[name]])
            g = g.dropna(subset=["cluster"])
            pt = per_target(g, "y_pred").dropna(subset=["spearman"])
            r["spearman"] = float(pt["spearman"].mean())
            r.update({k: float(v) for k, v in pair_stats(pairs(g, "y_pred")).items()
                      if k in ("sign_accuracy", "slope_dyhat_on_dy")})

        rows.append(r)

    per_fold = pd.DataFrame(rows)
    cols = [c for c in ["rmse", "mae", "ci", "pearson", "spearman",
                        "sign_accuracy", "slope_dyhat_on_dy"] if c in per_fold]
    summary = (per_fold.groupby(["set", "model"])[cols].agg(["mean", "std"]))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()

    for name in sets:
        print(f"\n{name}: mean +- std over {len(args.folds)} folds (single models)")
        print(f"{'model':<20}{'RMSE':>15}{'Pearson':>15}{'CI':>15}"
              + (f"{'Spearman':>15}{'slope':>15}" if PARENT[name] == name else ""))
        for m in MODELS:
            r = summary[(summary["set"] == name) & (summary["model"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            line = "".join(f"{r[c + '_mean']:>9.3f} ±{r[c + '_std']:>5.3f}"
                           for c in ["rmse", "pearson", "ci"])
            if PARENT[name] == name:
                line += "".join(f"{r[c + '_mean']:>9.3f} ±{r[c + '_std']:>5.3f}"
                                for c in ["spearman", "slope_dyhat_on_dy"])
            print(f"{LABELS[m]:<20}{line}")

    # ---------------- paired bootstrap over targets ----------------
    rng = np.random.default_rng(args.boot_seed)
    e2 = per_complex_e2(d)
    sp = per_target_spearman(d, labels)
    pairs_ab = [("full", "ligand"), ("full", "fp"), ("ligand", "fp")]
    boot_rows = []

    for name in sets:
        sub = e2[e2["set"] == name].copy()
        sub["cluster"] = sub["pdb_id"].map(labels[PARENT[name]])
        sub = sub.dropna(subset=["cluster"])

        for a, b in pairs_ab:
            mean, lo, hi = boot_diff_weighted(sub[b] - sub[a], sub["cluster"], args.n_boot, rng)
            boot_rows.append({"set": name, "metric": "mse_diff", "better": a, "worse": b,
                              "n_targets": sub["cluster"].nunique(), "diff": mean,
                              "lo": lo, "hi": hi})

        if PARENT[name] != name:
            continue

        s = sp[sp["set"] == name]
        for a, b in pairs_ab:
            mean, lo, hi = boot_diff_targets((s[a] - s[b]).to_numpy(), args.n_boot, rng)
            boot_rows.append({"set": name, "metric": "spearman_diff", "better": a, "worse": b,
                              "n_targets": len(s), "diff": mean, "lo": lo, "hi": hi})

    boot = pd.DataFrame(boot_rows)

    print(f"\nPaired bootstrap over targets ({args.n_boot} resamples, the same targets "
          f"for every model).")
    print("mse_diff: mean e^2 of the second model minus the first, so positive means the "
          "first is better.")
    print("spearman_diff: within-target Spearman of the first minus the second, so positive "
          "means the first ranks better.\n")
    print(f"{'set':<16}{'metric':<16}{'comparison':<34}{'diff':>8}{'95% CI':>20}")
    for _, r in boot.iterrows():
        comp = f"{LABELS[r['better']]} vs {LABELS[r['worse']]}"
        sig = "" if r["lo"] <= 0 <= r["hi"] else "  *"
        print(f"{r['set']:<16}{r['metric']:<16}{comp:<34}{r['diff']:>8.3f}"
              f"   [{r['lo']:>6.3f}, {r['hi']:>6.3f}]{sig}")
    print("\n* marks an interval that does not contain zero.")

    per_fold.to_csv(os.path.join(out_dir, "per_fold.csv"), index=False)
    summary.round(4).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    boot.round(4).to_csv(os.path.join(out_dir, "paired.csv"), index=False)
    dropped.to_csv(os.path.join(out_dir, "common_ids.csv"), index=False)

    print(f"\nSaved {out_dir}/per_fold.csv, summary.csv, paired.csv and common_ids.csv")


if __name__ == "__main__":
    main()
