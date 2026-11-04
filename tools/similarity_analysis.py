"""
How similar are validation and test complexes to the training set, and does
the model's error depend on it?

Uses the pairwise similarity matrices published with GEMS / CleanSplit
(Zenodo 10.5281/zenodo.17190227: pairwise_similarity_matrices.tar.gz and
pairwise_similarity_complexes.json):
    pairwise_similarity_complexes.json        PDB IDs in matrix order
    pairwise_similarity_matrix_tanimoto.npy   ligand Tanimoto similarity
    pairwise_similarity_matrix_tm.npy         protein TM-score
    pairwise_similarity_matrix_rmsd.npy       pocket-aligned ligand RMSD

For every complex of val, each CASF set and a random sample of train
(leave-self-out) it computes the similarity to the closest training complex:
    max_tanimoto, max_tm         best ligand / protein similarity
    max_search                   best Tanimoto + TM-score (score of the GEMS
                                 search algorithm)
    n_close                      number of training complexes that pass the
                                 CleanSplit filter criterion
                                 (Tanimoto > 0.9, or TM > 0.8 and
                                 Tanimoto + (1 - RMSD) > 0.8), label ignored
    knn5_pred                    GEMS search baseline on OUR train set:
                                 weighted mean pK of the 5 training complexes
                                 with the highest Tanimoto + TM-score

If val_predictions.csv / test_predictions.csv exist in the run folder, the
model's errors are joined and summarised by similarity.

Usage (from the repo root):
    python tools/similarity_analysis.py --sim_dir <folder with the matrices> \
        [--run_dir outputs/gnn_only_val15_seed42] [--train_sample 2000]

Outputs go to <run_dir>/similarity/:
    similarity_per_complex.csv, similarity_summary.csv,
    error_by_similarity.csv, similarity_hist.png, error_vs_similarity.png
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

TANIMOTO_T, TM_T, S_T = 0.9, 0.8, 0.8   # CleanSplit filter thresholds
TOP_N = 5


def find_file(sim_dir, name):
    for root, _, files in os.walk(sim_dir):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(f"{name} not found under {sim_dir}")


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def pearson(a, b):
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    return pearson(pd.Series(a).rank().values, pd.Series(b).rank().values)


def analyse_set(name, query_ids, train_pos, train_ids_arr, train_pk, idx, T, M, R, exclude_self):
    rows = []
    for pid in query_ids:
        i = idx[pid]
        tan = np.asarray(T[i])[train_pos].astype(np.float64)
        tm = np.asarray(M[i])[train_pos].astype(np.float64)
        rmsd = np.asarray(R[i])[train_pos].astype(np.float64)
        search = tan + tm
        s_score = tan + (1.0 - rmsd)
        if exclude_self:
            own = train_ids_arr == pid
            for arr in (tan, tm, search, s_score):
                arr[own] = -np.inf
        close = (tan > TANIMOTO_T) | ((tm > TM_T) & (s_score > S_T))
        top = np.argsort(-search)[:TOP_N]
        w = search[top]
        knn = float(np.average(train_pk[top], weights=w)) if np.all(np.isfinite(w)) and w.sum() > 0 \
            else float(np.mean(train_pk[top]))
        nn = top[0]
        rows.append({
            "set": name,
            "pdb_id": pid,
            "max_tanimoto": float(np.nanmax(tan)),
            "max_tm": float(np.nanmax(tm)),
            "max_search": float(search[nn]),
            "n_close": int(close.sum()),
            "nn_id": train_ids_arr[nn],
            "nn_pk": float(train_pk[nn]),
            "nn_tanimoto": float(tan[nn]),
            "nn_tm": float(tm[nn]),
            "knn5_pred": knn,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_dir", required=True, help="folder with the GEMS similarity matrices")
    ap.add_argument("--run_dir", default=None, help="output folder of a run (default: config.output_dir)")
    ap.add_argument("--train_sample", type=int, default=2000,
                    help="random training complexes for the train-vs-train reference (0 = skip)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from config import Config
    c = Config()
    run_dir = args.run_dir or c.output_dir
    out_dir = os.path.join(run_dir, "similarity")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(run_dir, "split_info.json")) as f:
        split = json.load(f)
    table = pd.read_csv(c.complexes_csv, dtype={"pdb_id": str}).set_index("pdb_id")

    with open(find_file(args.sim_dir, "pairwise_similarity_complexes.json")) as f:
        complexes = json.load(f)
    idx = {pid: k for k, pid in enumerate(complexes)}
    print(f"Similarity matrices: {len(complexes)} complexes")

    # Memory-mapped: only the rows that are needed are read from disk
    T = np.load(find_file(args.sim_dir, "pairwise_similarity_matrix_tanimoto.npy"), mmap_mode="r")
    M = np.load(find_file(args.sim_dir, "pairwise_similarity_matrix_tm.npy"), mmap_mode="r")
    R = np.load(find_file(args.sim_dir, "pairwise_similarity_matrix_rmsd.npy"), mmap_mode="r")
    print(f"Matrix shape {T.shape}, dtype {T.dtype}")

    train_ids = [i for i in split["train_ids"] if i in idx]
    missing = len(split["train_ids"]) - len(train_ids)
    if missing:
        print(f"Warning: {missing} train IDs are not in the similarity matrices and are ignored")
    train_ids_arr = np.array(train_ids)
    train_pos = np.array([idx[i] for i in train_ids])
    train_pk = table.loc[train_ids, "pk"].to_numpy(dtype=float)

    sets = {"val": split["val_ids"], **split["test_ids"]}
    if args.train_sample > 0:
        rng = np.random.default_rng(args.seed)
        n = min(args.train_sample, len(train_ids))
        sets = {"train (sample, self excluded)": sorted(rng.choice(train_ids, n, replace=False).tolist()), **sets}

    rows = []
    for name, ids in sets.items():
        ids_in = [i for i in ids if i in idx]
        if len(ids_in) < len(ids):
            print(f"Warning: {len(ids) - len(ids_in)} {name} IDs are not in the matrices")
        print(f"{name}: {len(ids_in)} complexes ...", flush=True)
        rows += analyse_set(name, ids_in, train_pos, train_ids_arr, train_pk, idx, T, M, R,
                            exclude_self=name.startswith("train"))
    df = pd.DataFrame(rows)
    df["y_true"] = table.loc[df["pdb_id"], "pk"].to_numpy(dtype=float)
    df["knn5_abs_err"] = (df["knn5_pred"] - df["y_true"]).abs()

    # Model predictions, if available
    preds = []
    for fname in ("val_predictions.csv", "test_predictions.csv"):
        p = os.path.join(run_dir, fname)
        if os.path.exists(p):
            preds.append(pd.read_csv(p, dtype={"pdb_id": str}))
        else:
            print(f"Note: {fname} not found, model errors for those sets are skipped")
    if preds:
        pr = pd.concat(preds).rename(columns={"test_set": "set"})[["set", "pdb_id", "y_pred"]]
        df = df.merge(pr, on=["set", "pdb_id"], how="left")
        df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
    else:
        df["y_pred"] = np.nan
        df["abs_err"] = np.nan

    df.to_csv(os.path.join(out_dir, "similarity_per_complex.csv"), index=False)

    # ---- Summary per set ----
    summ = []
    for name, g in df.groupby("set", sort=False):
        has = g["y_pred"].notna()
        summ.append({
            "set": name,
            "n": len(g),
            "median_max_tanimoto": g["max_tanimoto"].median(),
            "median_max_tm": g["max_tm"].median(),
            "median_max_search": g["max_search"].median(),
            "frac_with_close_train": (g["n_close"] > 0).mean(),
            "knn5_rmse": rmse(g["knn5_pred"], g["y_true"]),
            "knn5_r": pearson(g["knn5_pred"], g["y_true"]),
            "model_rmse": rmse(g.loc[has, "y_pred"], g.loc[has, "y_true"]) if has.any() else np.nan,
            "model_r": pearson(g.loc[has, "y_pred"], g.loc[has, "y_true"]) if has.any() else np.nan,
            "r_model_vs_knn5": pearson(g.loc[has, "y_pred"], g.loc[has, "knn5_pred"]) if has.any() else np.nan,
            "spearman_abs_err_vs_max_search": spearman(g.loc[has, "abs_err"], g.loc[has, "max_search"]) if has.any() else np.nan,
        })
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(out_dir, "similarity_summary.csv"), index=False)

    # ---- Model error by similarity bin (val + all test sets pooled per set) ----
    bins = [-np.inf, 1.0, 1.3, 1.6, np.inf]
    labels = ["<1.0", "1.0-1.3", "1.3-1.6", ">=1.6"]
    df["search_bin"] = pd.cut(df["max_search"], bins=bins, labels=labels, right=False)
    eb = (df[df["y_pred"].notna()]
          .groupby(["set", "search_bin"], sort=False, observed=True)
          .apply(lambda g: pd.Series({
              "n": len(g),
              "model_rmse": rmse(g["y_pred"], g["y_true"]),
              "knn5_rmse": rmse(g["knn5_pred"], g["y_true"]),
          }), include_groups=False)
          .reset_index())
    if len(eb):
        eb["n"] = eb["n"].astype(int)
    eb.to_csv(os.path.join(out_dir, "error_by_similarity.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Similarity to the training set (closest training complex) ===")
    print(summ[["set", "n", "median_max_tanimoto", "median_max_tm", "median_max_search",
                "frac_with_close_train"]].round(3).to_string(index=False))
    print("\n=== Search baseline (top-5 by Tanimoto + TM, our train) vs model ===")
    print(summ[["set", "knn5_rmse", "knn5_r", "model_rmse", "model_r", "r_model_vs_knn5",
                "spearman_abs_err_vs_max_search"]].round(3).to_string(index=False))
    if len(eb):
        print("\n=== Model RMSE by similarity to train (max Tanimoto + TM) ===")
        print(eb.round(3).to_string(index=False))

    # ---- Plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for col, ax in zip(["max_tanimoto", "max_tm", "max_search"], axes):
            for name, g in df.groupby("set", sort=False):
                ax.hist(g[col], bins=40, histtype="step", density=True, label=f"{name} (n={len(g)})")
            ax.set_xlabel(col)
            ax.set_ylabel("density")
        axes[0].legend(fontsize=8)
        fig.suptitle("Similarity to the closest training complex")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "similarity_hist.png"), dpi=150)

        sub = df[df["y_pred"].notna()]
        if len(sub):
            names = list(dict.fromkeys(sub["set"]))
            fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), squeeze=False)
            for name, ax in zip(names, axes[0]):
                g = sub[sub["set"] == name]
                ax.scatter(g["max_search"], g["abs_err"], s=8, alpha=0.5)
                ax.set_title(f"{name}\nSpearman {spearman(g['abs_err'], g['max_search']):.2f}")
                ax.set_xlabel("max Tanimoto + TM to train")
                ax.set_ylabel("|error|, pK")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "error_vs_similarity.png"), dpi=150)
    except ImportError:
        print("matplotlib not installed, plots skipped")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
