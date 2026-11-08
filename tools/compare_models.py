"""
Paired comparison of three models on exactly the same complexes.

    full    the GNN on complex graphs      outputs/<full>_fold{k}_seed{s}/test_predictions.csv
    ligand  the same GNN on ligand graphs  outputs/<ligand>_fold{k}_seed{s}/test_predictions.csv
    fp      Morgan fingerprints + boosting outputs/<full>_fold{k}_seed{s}/ligand_fp/predictions.csv

The three are trained on the same folds of the same split, so the comparison is
paired per complex. The headline result is the paired difference of mean
squared error with a cluster confidence interval; the three RMSEs are the
context that makes it readable. Within-target Spearman is the second,
independent criterion: it asks whether the advantage survives when the
comparison is made between ligands of one target.

What the script insists on:

  * one prediction per (model, fold, complex), and every kept complex
    predicted by all three models in every fold. The fingerprint baseline
    silently drops ligands RDKit cannot read, and a set that differs by a few
    complexes is not the same set. Duplicate rows stop the script.
  * every kept complex carries a target from CoreSet.dat, so nothing falls out
    of the cluster analysis unnoticed (--drop_untargeted to continue anyway).
  * a target enters the ranking comparison only if its Spearman is defined for
    every model in every fold, so all three are scored on the same targets.
  * the squared error of a complex is averaged over the folds, never the
    predictions: averaging predictions builds an ensemble, which is a
    different model. Ranking is combined the same way, per target per fold
    first and then over folds.

Uncertainty: a paired bootstrap over CASF targets (the protein clusters of
CoreSet.dat), not over complexes, because complexes of one target are not
independent. The resampled targets are drawn once per test set and reused by
every comparison, so the three differences are measured on the same draws and
their joint distribution is saved. The folds are held fixed, not resampled:
five is too few to resample, so the interval covers the sampling of targets
only, and that is how it should be described.

The independent subsets borrow the targets of their parent set and report no
within-target ranking, their groups being incomplete.

Usage (from the repo root):
    python tools/compare_models.py --coreset /path/CoreSet2016.dat \
                                   --coreset2013 /path/CoreSet2013.dat \
                                   [--full gnn_only_noH_amp] \
                                   [--ligand gnn_only_noH_amp_ligandonly] \
                                   [--folds 0 1 2 3 4] [--seed 42] [--n_boot 2000]

Writes per_fold.csv, summary.csv, paired.csv, common_ids.csv and
bootstrap_draws.csv into outputs/cv_summary_<full>/compare/.
"""
import os
import sys
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

from casf_ranking import read_coreset, per_target, pairs, pair_stats
from trainer import Trainer

# A model that predicts the same value for every ligand of a target has no
# defined Spearman there. That is a result, not a problem: the target then
# leaves the ranking comparison for every model and n_targets says so.
warnings.filterwarnings("ignore", message="An input array is constant")
warnings.filterwarnings("ignore", message="An input array is nearly constant")

MODELS = ["full", "ligand", "fp"]
LABELS = {"full": "GNN complex", "ligand": "GNN ligand only", "fp": "Morgan + boosting"}
COMPARISONS = [("full", "ligand"), ("full", "fp"), ("ligand", "fp")]

# Which CoreSet file gives the targets of each test set. The independent
# subsets are subsets of their parent set, so they inherit its targets.
PARENT = {"casf2016": "casf2016", "casf2016_indep": "casf2016",
          "casf2013": "casf2013", "casf2013_indep": "casf2013"}

# Comparisons decided on before seeing the numbers. They are reported
# uncorrected; everything else is a secondary family and gets Holm.
PRIMARY = {("casf2016", "mse_diff", "full", "ligand"),
           ("casf2016", "mse_diff", "full", "fp")}


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


def common_ids(d, folds):
    """
    Keep the complexes predicted exactly once by every model in every fold.

    Counting rows is not enough: a duplicated row of one model could make up
    for a complex another model is missing. The invariant the comparison needs
    is the set of (model, fold) keys, so that is what is checked.
    """
    keys = ["set", "pdb_id", "model", "fold"]
    dup = d[d.duplicated(keys, keep=False)]
    if len(dup):
        first = dup.sort_values(keys).head(6)[keys + ["y_pred"]]
        raise SystemExit("Duplicate predictions for the same model, fold and complex; "
                         "the comparison would not be paired.\n" + first.to_string(index=False))

    # With duplicates ruled out, every (model, fold) key of a complex is
    # distinct and comes from the expected universe, so counting the distinct
    # keys is the same test as comparing the sets, and needs no groupby.apply.
    expected = [(m, k) for m in MODELS for k in folds]
    d = d.assign(_key=d["model"] + "/f" + d["fold"].astype(str))
    have = d.groupby(["set", "pdb_id"])["_key"].nunique().rename("n_keys").reset_index()
    have["ok"] = have["n_keys"] == len(expected)

    present = set(zip(d["set"], d["pdb_id"], d["_key"]))
    rows = []
    for name, g in have.groupby("set"):
        bad = g[~g["ok"]]
        why = []
        for pdb_id in bad["pdb_id"].head(5):
            miss = [f"{m}/f{k}" for m, k in expected
                    if (name, pdb_id, f"{m}/f{k}") not in present]
            why.append(f"{pdb_id} (missing {','.join(miss)})")
        rows.append({"set": name, "n_common": int(g["ok"].sum()), "n_dropped": len(bad),
                     "dropped_ids": "; ".join(why)})

    good = set(map(tuple, have.loc[have["ok"], ["set", "pdb_id"]].to_numpy()))
    keep = d[[(s_, i) in good for s_, i in zip(d["set"], d["pdb_id"])]].drop(columns="_key")

    return keep.copy(), pd.DataFrame(rows)


def check_labels(d):
    """The same complex must carry the same label everywhere, or these are not the same data."""
    spread = d.groupby(["set", "pdb_id"])["y_true"].agg(lambda s: s.max() - s.min())
    worst = float(spread.max()) if len(spread) else 0.0

    if worst > 1e-4:
        bad = spread.idxmax()
        raise SystemExit(f"The label of {bad[1]} in {bad[0]} differs between models or folds "
                         f"(gap {worst:.3f} pK). The runs are not on the same data.")


def attach_targets(d, labels, drop_untargeted):
    """Give every row its CASF target, and say so loudly if some complex has none."""
    d = d.copy()
    d["cluster"] = [labels[PARENT[s]].get(i, None) for s, i in zip(d["set"], d["pdb_id"])]
    missing = d[d["cluster"].isna()]

    if len(missing):
        per_set = missing.groupby("set")["pdb_id"].nunique()
        ids = sorted(missing["pdb_id"].unique())
        msg = ("Complexes with no target in CoreSet.dat: "
               + ", ".join(f"{s} {n}" for s, n in per_set.items())
               + "\n  " + " ".join(ids[:20]) + (" ..." if len(ids) > 20 else ""))
        if not drop_untargeted:
            raise SystemExit(msg + "\nThey would silently leave the cluster analysis. "
                                   "Check the CoreSet files, or pass --drop_untargeted.")
        print("Warning: " + msg)
        d = d.dropna(subset=["cluster"])

    return d


def spearman_table(d, folds):
    """
    Mean over folds of the within-target Spearman, per model and target.

    A target is kept only where the Spearman is defined for every model in
    every fold: a target scored for one model and not for another would make
    the three columns incomparable and the count printed next to them wrong.
    """
    rows = []

    for (name, model, fold), g in d.groupby(["set", "model", "fold"]):
        pt = per_target(g, "y_pred")
        for _, r in pt.iterrows():
            rows.append({"set": name, "model": model, "fold": fold,
                         "cluster": r["cluster"], "n_in_group": r["n"],
                         "spearman": r["spearman"]})

    if not rows:
        return pd.DataFrame(columns=["set", "cluster", "n_in_group"] + MODELS)

    s = pd.DataFrame(rows)
    need = len(MODELS) * len(folds)
    ok = (s.dropna(subset=["spearman"]).groupby(["set", "cluster"])["spearman"].size() == need)
    ok = set(ok[ok].index)

    s = s[[(a, b) in ok for a, b in zip(s["set"], s["cluster"])]]
    size = s.groupby(["set", "cluster"])["n_in_group"].first()
    out = (s.groupby(["set", "model", "cluster"])["spearman"].mean()
           .unstack("model").reset_index())

    return out.merge(size.rename("n_in_group").reset_index(), on=["set", "cluster"])


def weighted_draws(values, groups, idx, order):
    """
    Bootstrap draws of a mean over complexes, resampling whole targets.

    The complexes of a target move together, so a draw is
    sum(target sums) / sum(target counts), never a mean of means.
    `idx` indexes `order`, the target list shared by every comparison.
    """
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    agg = df.groupby("g")["v"].agg(["sum", "count"]).reindex(order).fillna(0.0)
    sums, counts = agg["sum"].to_numpy(), agg["count"].to_numpy()

    return sums.sum() / counts.sum(), sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def target_draws(values, idx):
    """Bootstrap draws of a mean over targets (one value per target)."""
    v = np.asarray(values, dtype=float)

    return v.mean(), v[idx].mean(axis=1)


def summarize(point, draws):
    """Point estimate, percentile interval and the two-sided bootstrap p-value."""
    p = 2 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0)))

    return (float(point), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)),
            float(min(1.0, max(p, 1.0 / len(draws)))))


def holm(p):
    """Holm step-down adjustment; returns adjusted p-values in the input order."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    adj = np.empty(m)
    run = 0.0

    for rank, i in enumerate(np.argsort(p)):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)

    return adj


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
    ap.add_argument("--drop_untargeted", action="store_true",
                    help="continue when a complex has no target in CoreSet.dat")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    official = {k: v for k, v in [("casf2016", args.coreset),
                                  ("casf2013", args.coreset2013)] if v}
    if not official:
        raise SystemExit("Give at least one of --coreset / --coreset2013: the bootstrap "
                         "resamples CASF targets, which only those files define.")

    labels = {s: read_coreset(p) for s, p in official.items()}
    group_size = {s: lab.value_counts() for s, lab in labels.items()}
    out_dir = args.out_dir or os.path.join(args.outputs, f"cv_summary_{args.full}", "compare")
    os.makedirs(out_dir, exist_ok=True)

    d = load(args)
    check_labels(d)
    d, dropped = common_ids(d, args.folds)

    sets = [s for s in PARENT if s in set(d["set"]) and PARENT[s] in labels]
    skipped = sorted(set(d["set"]) - set(sets))
    d = d[d["set"].isin(sets)].copy()
    d = attach_targets(d, labels, args.drop_untargeted)

    # Complexes without a target leave with --drop_untargeted, so the count
    # printed below has to be the one the analysis actually uses.
    actual = d.groupby("set")["pdb_id"].nunique()
    dropped["n_common"] = (dropped["set"].map(actual)
                           .fillna(dropped["n_common"]).astype(int))

    print(f"Folds:  {args.folds}, seed {args.seed}")
    print("Models: " + ", ".join(f"{m} ({LABELS[m]})" for m in MODELS))
    print("\nComplexes used (one prediction from every model in every fold):")
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

        # Ranking only where the set owns its targets. The independent subsets
        # are scattered across the parent's targets, so their groups are
        # fragments and a within-target order means nothing there.
        if PARENT[name] == name:
            pt = per_target(g, "y_pred").dropna(subset=["spearman"])
            r["spearman"] = float(pt["spearman"].mean())
            r.update({k: float(v) for k, v in pair_stats(pairs(g, "y_pred")).items()
                      if k in ("sign_accuracy", "slope_dyhat_on_dy")})

        rows.append(r)

    per_fold = pd.DataFrame(rows)
    cols = [c for c in ["rmse", "mae", "ci", "pearson", "spearman",
                        "sign_accuracy", "slope_dyhat_on_dy"] if c in per_fold]
    summary = per_fold.groupby(["set", "model"])[cols].agg(["mean", "std"])
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

    print("\n± here is the spread of the five trained models, not a confidence interval "
          "over targets.\nPearson, CI and the slope have no paired interval: only the "
          "differences below are bootstrapped.")

    # ---------------- paired bootstrap over targets ----------------
    rng = np.random.default_rng(args.boot_seed)
    e2 = (d.assign(e2=(d["y_pred"] - d["y_true"]) ** 2)
          .groupby(["set", "cluster", "model", "pdb_id"])["e2"].mean()
          .unstack("model").reset_index())
    sp = spearman_table(d, args.folds)

    boot_rows, draw_cols = [], {}

    for name in sets:
        sub = e2[e2["set"] == name]
        order = sorted(sub["cluster"].unique())
        # One set of resampled targets per test set, reused by all three
        # comparisons, so their draws are directly comparable and can be
        # combined afterwards.
        idx = rng.integers(0, len(order), size=(args.n_boot, len(order)))

        for a, b in COMPARISONS:
            point, draws = weighted_draws(sub[b] - sub[a], sub["cluster"], idx, order)
            est, lo, hi, p = summarize(point, draws)
            boot_rows.append({"set": name, "metric": "mse_diff", "better": a, "worse": b,
                              "n_targets": len(order), "n": len(sub),
                              "diff": est, "lo": lo, "hi": hi, "p": p})
            draw_cols[f"{name}|mse_diff|{a}-{b}"] = draws

        if PARENT[name] != name:
            continue

        s = sp[sp["set"] == name]
        if s.empty:
            continue

        complete = s[s["n_in_group"] == s["cluster"].map(group_size[PARENT[name]])]
        for tag, t in (("spearman_diff", s), ("spearman_diff_complete", complete)):
            if t.empty:
                continue
            tidx = rng.integers(0, len(t), size=(args.n_boot, len(t)))
            for a, b in COMPARISONS:
                point, draws = target_draws((t[a] - t[b]).to_numpy(), tidx)
                est, lo, hi, p = summarize(point, draws)
                boot_rows.append({"set": name, "metric": tag, "better": a, "worse": b,
                                  "n_targets": len(t), "n": int(t["n_in_group"].sum()),
                                  "diff": est, "lo": lo, "hi": hi, "p": p})
                draw_cols[f"{name}|{tag}|{a}-{b}"] = draws

    boot = pd.DataFrame(boot_rows)
    boot["primary"] = [(r["set"], r["metric"], r["better"], r["worse"]) in PRIMARY
                       for _, r in boot.iterrows()]
    boot["p_holm"] = np.nan
    sec = ~boot["primary"]
    if sec.any():
        boot.loc[sec, "p_holm"] = holm(boot.loc[sec, "p"].to_numpy())

    print(f"\nPaired bootstrap over targets ({args.n_boot} resamples; one set of resampled "
          f"targets per\ntest set, shared by the three comparisons; the five folds are held "
          f"fixed, so the\ninterval covers the sampling of targets only).")
    print("mse_diff: mean e^2 of the second model minus the first, so positive means the "
          "first is better.")
    print("spearman_diff: within-target Spearman of the first minus the second. "
          "'_complete' uses\nonly targets holding all their CoreSet ligands.\n")
    print(f"{'set':<16}{'metric':<24}{'comparison':<34}{'n_t':>4}{'diff':>8}"
          f"{'95% CI':>20}{'p':>9}{'p_Holm':>9}")
    for _, r in boot.iterrows():
        comp = f"{LABELS[r['better']]} vs {LABELS[r['worse']]}"
        holm_s = "primary" if r["primary"] else f"{r['p_holm']:.3f}"
        print(f"{r['set']:<16}{r['metric']:<24}{comp:<34}{r['n_targets']:>4}{r['diff']:>8.3f}"
              f"   [{r['lo']:>6.3f}, {r['hi']:>6.3f}]{r['p']:>9.3f}{holm_s:>9}")

    print("\nThe two primary comparisons (full vs ligand and full vs fp on casf2016, "
          "mse_diff) were\nchosen before the numbers and are reported uncorrected. "
          "Every other row is a secondary\nfamily member and carries a Holm-adjusted "
          "p. Intervals are uncorrected 95% percentile\nbootstrap intervals, so read "
          "them next to p_Holm rather than on their own.")

    per_fold.to_csv(os.path.join(out_dir, "per_fold.csv"), index=False)
    summary.round(4).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    boot.round(4).to_csv(os.path.join(out_dir, "paired.csv"), index=False)
    dropped.to_csv(os.path.join(out_dir, "common_ids.csv"), index=False)
    pd.DataFrame(draw_cols).to_csv(os.path.join(out_dir, "bootstrap_draws.csv"), index=False)

    print(f"\nSaved {out_dir}/per_fold.csv, summary.csv, paired.csv, common_ids.csv "
          f"and bootstrap_draws.csv")


if __name__ == "__main__":
    main()
