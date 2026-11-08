"""
Ligand-only control that does not use a neural network:
Morgan fingerprints of the ligand + gradient boosting.

Purpose: an independent check of how much of the affinity can be predicted
from the ligand alone. If the GNN trained on ligand graphs only (graph_input
= "ligand") lands near this number, the ligand representation is not the
limiting factor; if it lands far below, the ligand branch is underfitting.

Uses exactly the train/val/test IDs of a finished run, read from its
split_info.json, so the comparison with that run is on the same split.

Usage (from the repo root, on Colab where scikit-learn is installed):
    python tools/ligand_fp_baseline.py outputs/gnn_only_noH_amp_fold0_seed42

Options:
    --radius 2 --bits 2048     Morgan fingerprint settings
    --counts                   use bond counts instead of a binary fingerprint
    --model hgb|rf             gradient boosting (default) or random forest
    --loss squared_error       matches the reported RMSE (default);
                               absolute_error is robust to label outliers

Writes into <run_dir>/ligand_fp/:
    metrics.csv          one row per set: n, RMSE, MAE, CI, Pearson
    predictions.csv      per-complex predictions for the paired analysis
    missing_ids.csv      IDs whose ligand could not be read, so a paired
                         comparison with the GNN is made on the intersection
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from config import Config
from cleansplit import load_complexes_table
from trainer import Trainer

RDLogger.DisableLog("rdApp.*")


def read_ligand(path):
    """Ligand from .mol2 or .sdf, without sanitization failures stopping us."""
    if path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(path, sanitize=False)
    else:
        mol = next(iter(Chem.SDMolSupplier(path, sanitize=False)), None)

    if mol is None:
        return None

    # Fingerprints need ring info and valences, not a full sanitization.
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)
    except Exception:
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
        except Exception:
            return None

    return mol


def fingerprints(ids, table, data_dir, gen, counts, bits):
    """Matrix of fingerprints, plus the IDs that could be read."""
    paths = table.set_index("pdb_id")["ligand_file"].to_dict()
    rows, kept, failed = [], [], []

    for pdb_id in ids:
        rel = paths.get(pdb_id)
        mol = read_ligand(os.path.join(data_dir, rel)) if rel else None

        if mol is None:
            failed.append(pdb_id)
            continue

        if counts:
            fp = gen.GetCountFingerprintAsNumPy(mol)
        else:
            fp = gen.GetFingerprintAsNumPy(mol)

        rows.append(fp.astype(np.float32))
        kept.append(pdb_id)

    x = np.vstack(rows) if rows else np.empty((0, bits), dtype=np.float32)

    return x, kept, failed


def metrics(y_true, y_pred):
    e = y_pred - y_true
    r = np.corrcoef(y_pred, y_true)[0, 1] if len(y_true) > 1 else np.nan

    return {
        "n": len(y_true),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "ci": float(Trainer.concordance_index(y_true, y_pred)),
        "pearson": float(r),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="folder of a finished run (holds split_info.json)")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--bits", type=int, default=2048)
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--model", choices=["hgb", "rf"], default="hgb")
    ap.add_argument("--loss", choices=["squared_error", "absolute_error"],
                    default="squared_error",
                    help="hgb: squared_error matches the reported RMSE, "
                         "absolute_error is more robust to label outliers")
    ap.add_argument("--max_iter", type=int, default=1200, help="hgb: most trees to try")
    ap.add_argument("--step", type=int, default=50, help="hgb: trees added per val check")
    ap.add_argument("--patience", type=int, default=4, help="hgb: val checks without a gain")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    c = Config()
    run_dir = args.run_dir or c.output_dir
    seed = args.seed if args.seed is not None else c.seed

    with open(os.path.join(run_dir, "split_info.json")) as f:
        split = json.load(f)

    table = load_complexes_table(c.complexes_csv)
    pk = table.set_index("pdb_id")["pk"].to_dict()

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=args.radius, fpSize=args.bits)

    sets = {"train": split["train_ids"], "val": split["val_ids"]}
    sets.update(split["test_ids"])

    print(f"Run:         {run_dir}")
    print(f"Fingerprint: Morgan r={args.radius}, {args.bits} bits, "
          f"{'counts' if args.counts else 'binary'}")

    x, y, ids, missing = {}, {}, {}, []
    for name, set_ids in sets.items():
        xi, kept, failed = fingerprints(set_ids, table, c.data_dir, gen, args.counts, args.bits)
        x[name] = xi
        ids[name] = kept
        y[name] = np.array([pk[i] for i in kept], dtype=np.float64)
        missing += [{"set": name, "pdb_id": i} for i in failed]
        print(f"{name:<16}{len(kept)} of {len(set_ids)} complexes"
              + (f", {len(failed)} ligands unreadable" if failed else ""))

    if args.model == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        def new_model(max_iter, warm_start=False):
            return HistGradientBoostingRegressor(
                loss=args.loss, max_iter=max_iter, learning_rate=0.06,
                max_leaf_nodes=63, min_samples_leaf=max(1, min(20, len(y["train"]) // 10)),
                l2_regularization=1.0, early_stopping=False,
                warm_start=warm_start, random_state=seed,
            )

        # Number of trees chosen on the same validation set the GNN uses to
        # pick its best epoch, so neither model gets a free look at the tests.
        print(f"Model:       HistGradientBoostingRegressor, loss {args.loss} "
              f"(trees chosen on val, up to {args.max_iter})")

        probe = new_model(args.step, warm_start=True)
        best_rmse, best_iter = np.inf, args.step

        for n_iter in range(args.step, args.max_iter + 1, args.step):
            probe.set_params(max_iter=n_iter)
            probe.fit(x["train"], y["train"])
            v = float(np.sqrt(np.mean((probe.predict(x["val"]) - y["val"]) ** 2)))
            print(f"  trees {n_iter:>5}  val RMSE {v:.4f}" + ("  [best]" if v < best_rmse else ""))

            if v < best_rmse:
                best_rmse, best_iter = v, n_iter
            elif n_iter - best_iter >= args.patience * args.step:
                break

        print(f"Best on val: {best_iter} trees, RMSE {best_rmse:.4f}")
        model = new_model(best_iter)
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed,
        )
        print(f"Model:       {model.__class__.__name__}")

    print("Fitting on train ...")
    model.fit(x["train"], y["train"])

    rows, preds = [], []
    for name in sets:
        p = model.predict(x[name])
        rows.append({"set": name, **metrics(y[name], p)})
        preds += [{"set": name, "pdb_id": i, "y_true": t, "y_pred": float(v)}
                  for i, t, v in zip(ids[name], y[name], p)]

    out_dir = os.path.join(run_dir, "ligand_fp")
    os.makedirs(out_dir, exist_ok=True)

    m = pd.DataFrame(rows)
    m.insert(1, "model", args.model)
    m.insert(2, "loss", args.loss if args.model == "hgb" else "squared_error")
    m["n_in_split"] = [len(sets[r["set"]]) for r in rows]
    m.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    pd.DataFrame(preds).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # A ligand this script cannot read is evaluated by the GNN but not here, so
    # any paired comparison has to be made on the intersection of the IDs.
    pd.DataFrame(missing, columns=["set", "pdb_id"]).to_csv(
        os.path.join(out_dir, "missing_ids.csv"), index=False)

    print("\n" + "=" * 62)
    print("LIGAND-ONLY FINGERPRINT BASELINE (pK scale)")
    print(f"{'set':<16}{'n':>5}{'RMSE':>9}{'MAE':>9}{'CI':>9}{'Pearson':>10}")
    for r in rows:
        print(f"{r['set']:<16}{r['n']:>5}{r['rmse']:>9.4f}{r['mae']:>9.4f}"
              f"{r['ci']:>9.4f}{r['pearson']:>10.4f}")
    print("=" * 62)

    if missing:
        n_test = sum(1 for r in missing if r["set"] not in ("train", "val"))
        print(f"\n{len(missing)} ligands could not be read ({n_test} in a test set); "
              f"their IDs are in missing_ids.csv.")
        print("Compare with the GNN on the intersection of pdb_id, not on both "
              "sets as they are.")

    print(f"\nSaved {out_dir}/metrics.csv, predictions.csv and missing_ids.csv")


if __name__ == "__main__":
    main()
