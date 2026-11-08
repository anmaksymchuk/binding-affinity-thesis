"""
Summary of a 5-fold cross-validation run: the numbers that go into the thesis.

Reads the fold folders of one configuration and reports

    1. per-fold metrics on every test set, and mean +- std over folds
       (this is what GEMS Fig. 3 reports: the mean of single models);
    2. the ensemble of the fold models on each test set, from the mean of
       their predictions. An ensemble is a different model from a single one,
       so it is reported separately and never mixed into the mean above;
    3. out-of-fold (OOF) validation metrics: the folds' validation parts are
       disjoint, so pooling their predictions covers the whole train pool
       once, each complex predicted by the model that did not train on it;
    4. published reference numbers on CASF-2016 for the same protocol.

Usage (from the repo root):
    python tools/summarize_folds.py [--prefix gnn_only_noH_amp] [--seed 42]
                                    [--folds 0 1 2 3 4] [--out_dir outputs/cv_summary]

Writes per_fold.csv, summary.csv, ensemble.csv and oof.csv into --out_dir.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import torch

from trainer import Trainer

# CleanSplit numbers on CASF-2016 from Graber et al. 2025, Fig. 3
# (mean over their 5 single CV models), for the comparison table.
PUBLISHED = [
    ("GEMS (with language models)", 1.308, 0.803),
    ("GenScore", 1.362, 0.780),
    ("Pafnucy", 1.484, 0.746),
    ("search top-5 (shortcut)", 1.648, 0.653),
]

METRICS = ["rmse", "mae", "ci", "pearson"]


def metrics_of(y_true, y_pred):
    return {
        "n": len(y_true),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "ci": float(Trainer.concordance_index(y_true, y_pred)),
        "pearson": float(np.corrcoef(y_pred, y_true)[0, 1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="gnn_only_noH_amp",
                    help="run folder name without _fold<k>_seed<s>")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.outputs, f"cv_summary_{args.prefix}")
    os.makedirs(out_dir, exist_ok=True)

    rows, test_preds, val_preds, splits = [], [], [], {}

    for k in args.folds:
        run = os.path.join(args.outputs, f"{args.prefix}_fold{k}_seed{args.seed}")

        if not os.path.isdir(run):
            raise SystemExit(f"{run} not found. Check --prefix, --seed and --folds.")

        m = torch.load(os.path.join(run, "test_metrics.pt"), weights_only=False)
        for name, met in m.items():
            rows.append({"fold": k, "set": name, **{c: met[c] for c in ["n"] + METRICS}})

        t = pd.read_csv(os.path.join(run, "test_predictions.csv"), dtype={"pdb_id": str})
        t["fold"] = k
        test_preds.append(t)

        v_path = os.path.join(run, "val_predictions.csv")
        if os.path.exists(v_path):
            v = pd.read_csv(v_path, dtype={"pdb_id": str})
            v["fold"] = k
            val_preds.append(v)

        with open(os.path.join(run, "split_info.json")) as f:
            info = json.load(f)
        splits[k] = info
        if info.get("fold") != k or info.get("val_mode") != "gems_fold":
            raise SystemExit(f"{run}: split_info says val_mode={info.get('val_mode')}, "
                             f"fold={info.get('fold')}, expected gems_fold {k}.")

    per_fold = pd.DataFrame(rows)
    sets = list(dict.fromkeys(per_fold["set"]))

    # The folds must be the same experiment, differing only in the fold.
    graph_inputs = {s.get("graph_input", "complex") for s in splits.values()}
    if len(graph_inputs) > 1:
        raise SystemExit(f"Folds disagree on graph_input: {graph_inputs}")

    val_sets = [set(s["val_ids"]) for s in splits.values()]
    overlap = sum(len(a & b) for i, a in enumerate(val_sets) for b in val_sets[i + 1:])

    # ---------------- 1. mean +- std over folds ----------------
    summary = (per_fold.groupby("set")[METRICS]
               .agg(["mean", "std"]).reindex(sets).round(4))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "n_folds", len(args.folds))
    summary.insert(1, "n", per_fold.groupby("set")["n"].first().reindex(sets))

    # ---------------- 2. ensemble of the fold models ----------------
    ens_rows = []
    all_test = pd.concat(test_preds)
    for name in sets:
        sub = all_test[all_test["test_set"] == name]
        g = sub.groupby("pdb_id").agg(y_true=("y_true", "first"),
                                      y_true_spread=("y_true", lambda s: s.max() - s.min()),
                                      y_pred=("y_pred", "mean"),
                                      n_models=("y_pred", "size"))

        # The same complex must carry the same label in every fold, otherwise
        # these are not the same test set and averaging them means nothing.
        if float(g["y_true_spread"].max()) > 1e-4:
            raise SystemExit(
                f"{name}: the label of the same complex differs between folds "
                f"(largest gap {g['y_true_spread'].max():.3f} pK). The folds were "
                "run on different data; check complexes.csv and the graph cache.")

        if g["n_models"].nunique() != 1:
            print(f"Warning: {name} has complexes predicted by different "
                  f"numbers of folds ({sorted(g['n_models'].unique())})")
        ens_rows.append({"set": name, **metrics_of(g["y_true"].to_numpy(),
                                                   g["y_pred"].to_numpy())})
    ensemble = pd.DataFrame(ens_rows)

    # ---------------- 3. out-of-fold validation ----------------
    oof = pd.DataFrame()
    if val_preds:
        v = pd.concat(val_preds)
        dup = int(v["pdb_id"].duplicated().sum())
        oof = pd.DataFrame([{"set": "val_oof", **metrics_of(v["y_true"].to_numpy(),
                                                            v["y_pred"].to_numpy()),
                             "duplicated_ids": dup}])

    # ---------------- output ----------------
    per_fold.to_csv(os.path.join(out_dir, "per_fold.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "summary.csv"))
    ensemble.to_csv(os.path.join(out_dir, "ensemble.csv"), index=False)
    if not oof.empty:
        oof.to_csv(os.path.join(out_dir, "oof.csv"), index=False)

    print(f"Runs:        {args.prefix}_fold{{{','.join(str(k) for k in args.folds)}}}"
          f"_seed{args.seed}")
    print(f"Graph input: {graph_inputs.pop()}")
    print(f"Validation parts disjoint: {'yes' if overlap == 0 else f'NO ({overlap} shared IDs)'}")

    print("\nPer fold, RMSE:")
    print(f"{'set':<18}" + "".join(f"{'f' + str(k):>9}" for k in args.folds))
    for name in sets:
        sub = per_fold[per_fold["set"] == name].set_index("fold")
        print(f"{name:<18}" + "".join(f"{sub.loc[k, 'rmse']:>9.3f}" for k in args.folds))

    print(f"\nMean +- std over {len(args.folds)} folds (single models):")
    print(f"{'set':<18}{'n':>5}{'RMSE':>16}{'MAE':>16}{'CI':>16}{'Pearson':>16}")
    for name in sets:
        r = summary.loc[name]
        cells = "".join(f"{r[m + '_mean']:>9.3f} ±{r[m + '_std']:>5.3f}" for m in METRICS)
        print(f"{name:<18}{int(r['n']):>5}{cells}")

    print("\nEnsemble of the fold models (mean of predictions, a different model):")
    print(f"{'set':<18}{'n':>5}{'RMSE':>9}{'MAE':>9}{'CI':>9}{'Pearson':>10}")
    for r in ens_rows:
        print(f"{r['set']:<18}{r['n']:>5}{r['rmse']:>9.3f}{r['mae']:>9.3f}"
              f"{r['ci']:>9.3f}{r['pearson']:>10.3f}")

    if not oof.empty:
        r = oof.iloc[0]
        print(f"\nOut-of-fold validation ({int(r['n'])} complexes, each predicted by the "
              f"model that did not train on it):")
        print(f"  RMSE {r['rmse']:.3f}  MAE {r['mae']:.3f}  CI {r['ci']:.3f}  "
              f"Pearson {r['pearson']:.3f}"
              + (f"  [{int(r['duplicated_ids'])} duplicated IDs]"
                 if r["duplicated_ids"] else ""))

    main_set = "casf2016"
    if main_set in sets:
        r = summary.loc[main_set]
        print(f"\nCASF-2016 against published CleanSplit numbers "
              f"(mean of 5 single models, Graber et al. 2025 Fig. 3):")
        print(f"{'model':<30}{'RMSE':>9}{'Pearson':>10}")
        print(f"{'this work (GNN, no LM)':<30}{r['rmse_mean']:>9.3f}{r['pearson_mean']:>10.3f}")
        for name, rmse, pear in PUBLISHED:
            print(f"{name:<30}{rmse:>9.3f}{pear:>10.3f}")

    print(f"\nSaved {out_dir}/per_fold.csv, summary.csv, ensemble.csv"
          + (", oof.csv" if not oof.empty else ""))


if __name__ == "__main__":
    main()
