"""
Within-target ranking on CASF (ranking power) and cluster-level statistics.

CASF-2016 groups its 285 complexes into 57 targets with 5 ligands each and
CASF-2013 its 195 complexes into 65 targets with 3 ligands each. Ranking
power asks whether a model orders the ligands of one target correctly, which
the global Pearson R / RMSE do not show: a model can get the target-level
average right and still rank the ligands of a target poorly, or the reverse.

Clusters. If the official CoreSet.dat files are given (--coreset for
CASF-2016, --coreset2013 for CASF-2013; the target column is found by a header
named "target"/"cluster", else the last integer column), they are used.
Without them the split is an approximate reconstruction, so do not compare the
numbers with published CASF ranking benchmarks. Otherwise clusters are rebuilt from a GEMS
protein similarity matrix: sequence identity (pairwise_similarity_matrix_seqid,
which matches how CASF defines its targets) if it is there, else TM-score
(--cluster_by to force one). Connected components of similarity > t among the
complexes of the set, with t chosen on a grid so that as many components as
possible have the official size (5 or 3); average-linkage clustering into the
official number of targets is tried too and the better split is kept. The
script prints the size distribution so you can see how clean it is. (TM-score
alone fragments CASF-2016 into ~140 groups, so sequence identity is preferred.) Cluster
labels are always computed on the FULL set (all 285 / 195 IDs, from
complexes.csv), so complexes without a prediction (e.g. failed graphs) leave
incomplete groups instead of shifting the clustering; they are listed.

Reads <run_dir>/similarity/similarity_per_complex.csv (from
tools/similarity_analysis.py), which has y_true, the model's y_pred and the
search baseline knn5_pred, and reports for both predictors:
  1. Per-target Spearman and Kendall, averaged over targets (CASF ranking
     power), with a bootstrap 95% CI over targets; high-level success rate
     (whole order right) and low-level success rate (best ligand predicted
     best), as in CASF-2013; paired bootstrap of the Spearman difference
     model - search on the same resampled targets.
  2. Pairwise differences within a target: error of predicted dpK,
     (yhat_i - yhat_j) - (y_i - y_j), sign accuracy and slope of predicted on
     true dpK, by |true dpK| bin. A target-wide offset cancels here.
     2b: the same by ligand Tanimoto of the pair (needs --sim_dir); the row
     Tanimoto >= 0.7 x |dpK| >= 2 holds activity-cliff-like pairs.
  3. MSE decomposition into a per-target offset (mean error of the target)
     and the error left within targets.
  4. Ranking vs the pK range of the target.
  5. Cluster bootstrap (resampling targets) for RMSE and for the paired
     difference mean(e_search^2 - e_model^2), also for the indep subsets using
     the parent set's targets.

Usage (from the repo root):
    python tools/casf_ranking.py --sim_dir <folder with the GEMS matrices> \
        [--run_dir outputs/gnn_only_val15_seed42] [--coreset CoreSet.dat]

Outputs go to <run_dir>/similarity/ranking/.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

GROUP_SIZE = {"casf2016": 5, "casf2013": 3}
INDEP_PARENT = {"casf2016_indep": "casf2016", "casf2013_indep": "casf2013"}
PREDICTORS = {"model": "y_pred", "search": "knn5_pred"}
DPK_BINS = [0.0, 1.0, 2.0, np.inf]
DPK_LABELS = ["<1", "1-2", ">=2"]


def find_file(sim_dir, name):
    for root, _, files in os.walk(sim_dir):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(f"{name} not found under {sim_dir}")


def n_exact(labels, k):
    sizes = pd.Series(labels).value_counts()
    return int((sizes == k).sum())


def cluster_by_matrix(ids, idx, M, k, name):
    pos = np.array([idx[i] for i in ids])
    sub = np.nan_to_num(np.asarray(M[np.ix_(pos, pos)], dtype=np.float64), nan=0.0)
    if np.nanmax(sub) > 1.5:          # percent -> fraction
        sub = sub / 100.0
    tm = (sub + sub.T) / 2
    np.fill_diagonal(tm, 1.0)
    best = None
    for t in np.round(np.arange(0.20, 0.995, 0.01), 2):
        _, lab = connected_components(csr_matrix(tm > t), directed=False)
        score = n_exact(lab, k)
        if best is None or score > best[0]:
            best = (score, lab, f"{name}, connected components at > {t:.2f}")
    dist = np.clip(1.0 - tm, 0, None)
    np.fill_diagonal(dist, 0.0)
    lab = fcluster(linkage(squareform(dist, checks=False), "average"),
                   t=len(ids) // k, criterion="maxclust")
    score = n_exact(lab, k)
    if score > best[0]:
        best = (score, lab, f"{name}, average linkage into {len(ids) // k} clusters")
    return pd.Series(best[1], index=ids), best[2]


def read_coreset(path):
    """PDB code -> target id from a CASF CoreSet.dat (whitespace separated)."""
    col, rows = None, []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                names = [x.lower() for x in line.lstrip("#").split()]
                for key in ("target", "cluster"):
                    if key in names:
                        col = names.index(key)
                continue
            parts = line.split()
            if col is not None and col < len(parts):
                val = parts[col]
            else:
                val = next(x for x in reversed(parts) if x.lstrip("-").isdigit())
            rows.append((parts[0].lower(), val))
    return pd.Series(dict(rows))


def boot_ci(values_fn, groups, n_boot, rng):
    groups = np.asarray(groups)
    vals = []
    for _ in range(n_boot):
        vals.append(values_fn(rng.choice(groups, len(groups), replace=True)))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def per_target(g, col):
    rows = []
    for c, sub in g.groupby("cluster"):
        r = {"cluster": c, "n": len(sub), "pk_range": sub["y_true"].max() - sub["y_true"].min(),
             "mean_error": (sub[col] - sub["y_true"]).mean()}
        if len(sub) >= 3:
            r["spearman"] = stats.spearmanr(sub[col], sub["y_true"])[0]
            r["kendall"] = stats.kendalltau(sub[col], sub["y_true"])[0]
            r["low_level_success"] = float(sub[col].idxmax() == sub["y_true"].idxmax())
            r["high_level_success"] = float((sub.sort_values("y_true")[col].diff().dropna() > 0).all())
        else:
            r["spearman"] = r["kendall"] = r["low_level_success"] = r["high_level_success"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def pairs(g, col, T=None, idx=None):
    rows = []
    for c, sub in g.groupby("cluster"):
        y, p, ids = sub["y_true"].to_numpy(), sub[col].to_numpy(), sub["pdb_id"].to_numpy()
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                tan = np.nan
                if T is not None and ids[i] in idx and ids[j] in idx:
                    tan = float(T[idx[ids[i]], idx[ids[j]]])
                rows.append({"cluster": c, "id_i": ids[i], "id_j": ids[j], "tanimoto": tan,
                             "dy": y[i] - y[j], "dyhat": p[i] - p[j]})
    return pd.DataFrame(rows)


def pair_stats(sub, slope=True):
    """Slope only over all pairs: inside a |dpK| bin the range of dpK is cut and the slope is biased."""
    nz = sub[sub["dy"] != 0]
    return {"n_pairs": len(sub),
            "rmse_dpk": np.sqrt(np.mean((sub["dyhat"] - sub["dy"]) ** 2)) if len(sub) else np.nan,
            "sign_accuracy": (np.sign(nz["dyhat"]) == np.sign(nz["dy"])).mean() if len(nz) else np.nan,
            "slope_dyhat_on_dy": np.polyfit(sub["dy"], sub["dyhat"], 1)[0] if slope and len(sub) >= 5 else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_dir", default=None, help="folder with the GEMS similarity matrices")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--coreset", default=None, help="optional CASF-2016 CoreSet.dat with official targets")
    ap.add_argument("--coreset2013", default=None, help="optional CASF-2013 CoreSet.dat with official targets")
    ap.add_argument("--cluster_by", choices=["auto", "seqid", "tm"], default="auto",
                    help="matrix for rebuilding CASF targets (auto: seqid if present, else tm)")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from config import Config
    c = Config()
    run_dir = args.run_dir or c.output_dir
    out = os.path.join(run_dir, "similarity", "ranking")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    d = pd.read_csv(os.path.join(run_dir, "similarity", "similarity_per_complex.csv"), dtype={"pdb_id": str})
    d = d[d["set"].isin(list(GROUP_SIZE) + list(INDEP_PARENT))]

    # Full ID lists of the CASF sets (including complexes without a prediction)
    full = {}
    if os.path.exists(c.complexes_csv):
        table = pd.read_csv(c.complexes_csv, dtype={"pdb_id": str})
        for s in GROUP_SIZE:
            if f"in_{s}" in table.columns:
                full[s] = sorted(table.loc[table[f"in_{s}"].astype(bool), "pdb_id"])
    for s in GROUP_SIZE:
        if s not in full:
            full[s] = sorted(d.loc[d["set"] == s, "pdb_id"])
            print(f"Note: full {s} ID list not found, clustering only the {len(full[s])} complexes with predictions")

    # ---- Clusters ----
    clusters = {}
    M = idx = T = None
    if args.sim_dir is not None:
        with open(find_file(args.sim_dir, "pairwise_similarity_complexes.json")) as f:
            idx = {pid: n for n, pid in enumerate(json.load(f))}
        T = np.load(find_file(args.sim_dir, "pairwise_similarity_matrix_tanimoto.npy"), mmap_mode="r")
    official = {"casf2016": args.coreset, "casf2013": args.coreset2013}
    for s, k in GROUP_SIZE.items():
        if official[s]:
            lab = read_coreset(official[s])
            how = f"official targets from {official[s]}"
            not_found = [i for i in full[s] if i not in lab.index]
            if not_found:
                print(f"  Warning: {len(not_found)} {s} IDs not in {official[s]}: {not_found[:10]}")
            lab = lab.reindex(full[s]).dropna()
        else:
            if M is None:
                if args.sim_dir is None:
                    sys.exit("--sim_dir is needed to rebuild the CASF targets from a similarity matrix")
                mname = args.cluster_by
                if mname == "auto":
                    try:
                        find_file(args.sim_dir, "pairwise_similarity_matrix_seqid.npy")
                        mname = "seqid"
                    except FileNotFoundError:
                        mname = "tm"
                M = np.load(find_file(args.sim_dir, f"pairwise_similarity_matrix_{mname}.npy"), mmap_mode="r")
            ids = [i for i in full[s] if i in idx]
            lab, how = cluster_by_matrix(ids, idx, M, k, mname)
        clusters[s] = lab
        sizes = lab.value_counts().value_counts().sort_index()
        print(f"{s}: {len(lab)} complexes -> {lab.nunique()} targets ({how}); "
              f"targets by size: { {int(a): int(b) for a, b in sizes.items()} }; expected {len(full[s]) // k} of size {k}")
        have = set(d.loc[d["set"] == s, "pdb_id"])
        missing = [i for i in full[s] if i not in have]
        if missing:
            print(f"  without prediction: " + ", ".join(f"{i} (target {lab.get(i)})" for i in missing))

    crows = []
    for s, lab in clusters.items():
        have = set(d.loc[d["set"] == s, "pdb_id"])
        for pid, cl in lab.items():
            crows.append({"set": s, "pdb_id": pid, "cluster": cl, "has_prediction": pid in have})
    pd.DataFrame(crows).to_csv(os.path.join(out, "casf_clusters.csv"), index=False)

    d["cluster"] = [clusters[INDEP_PARENT.get(s, s)].get(p) for s, p in zip(d["set"], d["pdb_id"])]
    d = d[d["y_pred"].notna() & d["cluster"].notna()]

    # ---- 1. Ranking power ----
    rank_rows, per_target_all, pair_rows, tan_rows, dec_rows = [], [], [], [], []
    TAN_BINS, TAN_LABELS = [0.0, 0.4, 0.7, 1.01], ["<0.4", "0.4-0.7", ">=0.7"]
    for s in GROUP_SIZE:
        g = d[d["set"] == s]
        sizes = g.groupby("cluster").size().value_counts().sort_index()
        size_str = ", ".join(f"{int(n)}x{int(sz)}" for sz, n in sizes.items())
        pts = {}
        for name, col in PREDICTORS.items():
            pt = per_target(g, col).assign(set=s, predictor=name)
            per_target_all.append(pt)
            ok = pt.dropna(subset=["spearman"]).set_index("cluster")
            pts[name] = ok
            lo, hi = boot_ci(lambda cs: ok.loc[cs, "spearman"].mean(), ok.index, args.n_boot, rng)
            rank_rows.append({
                "set": s, "predictor": name, "n_targets_used": len(ok), "groups_with_predictions": size_str,
                "mean_spearman": ok["spearman"].mean(), "spearman_ci_low": lo, "spearman_ci_high": hi,
                "mean_kendall": ok["kendall"].mean(),
                "high_level_success": ok["high_level_success"].mean(),
                "low_level_success": ok["low_level_success"].mean(),
                "share_targets_spearman_gt_0.5": (ok["spearman"] > 0.5).mean(),
                "global_pearson": np.corrcoef(g[col], g["y_true"])[0, 1],
            })

            # 2. Pairwise dpK
            pr = pairs(g, col, T, idx)
            pr["bin"] = pd.cut(pr["dy"].abs(), DPK_BINS, labels=DPK_LABELS, right=False)
            pair_rows.append({"set": s, "predictor": name, "dy_bin": "all", **pair_stats(pr)})
            for b, sub in pr[pr["dy"] != 0].groupby("bin", observed=True):
                pair_rows.append({"set": s, "predictor": name, "dy_bin": b, **pair_stats(sub, slope=False)})

            # 2b. By ligand similarity of the pair
            if pr["tanimoto"].notna().any():
                pr["tan_bin"] = pd.cut(pr["tanimoto"], TAN_BINS, labels=TAN_LABELS, right=False)
                for tb, sub in pr.groupby("tan_bin", observed=True):
                    tan_rows.append({"set": s, "predictor": name, "tanimoto": tb, "dy_bin": "all", **pair_stats(sub)})
                    for b, sub2 in sub[sub["dy"] != 0].groupby("bin", observed=True):
                        tan_rows.append({"set": s, "predictor": name, "tanimoto": tb, "dy_bin": b,
                                         **pair_stats(sub2, slope=False)})

            # 3. MSE decomposition: per-target offset + within-target error
            e = g[col] - g["y_true"]
            off = e.groupby(g["cluster"]).transform("mean")
            dec_rows.append({
                "set": s, "predictor": name, "rmse": np.sqrt(np.mean(e ** 2)),
                "rmse_target_offset": np.sqrt(np.mean(off ** 2)),
                "rmse_within_target": np.sqrt(np.mean((e - off) ** 2)),
                "share_mse_target_offset": np.mean(off ** 2) / np.mean(e ** 2),
                "std_true_within_target": np.sqrt(np.mean((g["y_true"] - g.groupby("cluster")["y_true"].transform("mean")) ** 2)),
                "std_pred_within_target": np.sqrt(np.mean((g[col] - g.groupby("cluster")[col].transform("mean")) ** 2)),
            })

        # paired bootstrap of the Spearman difference on the same targets
        common = pts["model"].index.intersection(pts["search"].index)
        diff = pts["model"].loc[common, "spearman"] - pts["search"].loc[common, "spearman"]
        lo, hi = boot_ci(lambda cs: diff.loc[cs].mean(), common, args.n_boot, rng)
        rank_rows.append({"set": s, "predictor": "model - search", "n_targets_used": len(common),
                          "groups_with_predictions": size_str,
                          "mean_spearman": diff.mean(), "spearman_ci_low": lo, "spearman_ci_high": hi,
                          "mean_kendall": (pts["model"].loc[common, "kendall"] - pts["search"].loc[common, "kendall"]).mean()})

    t_rank = pd.DataFrame(rank_rows)
    t_target = pd.concat(per_target_all, ignore_index=True)
    t_pairs = pd.DataFrame(pair_rows)
    t_tan = pd.DataFrame(tan_rows)
    t_dec = pd.DataFrame(dec_rows)
    print("\n=== 1. Ranking power: Spearman / Kendall within targets, averaged over targets ===")
    print("groups_with_predictions: number x size of the target groups that have predictions")
    print(t_rank.round(3).to_string(index=False))
    print("\n=== 2. Pairwise dpK within targets: (yhat_i - yhat_j) - (y_i - y_j) ===")
    print(t_pairs.round(3).to_string(index=False))
    if len(t_tan):
        print("\n=== 2b. Pairwise dpK by ligand Tanimoto of the pair (>=0.7 with |dpK| >= 2: cliff-like) ===")
        print(t_tan.round(3).to_string(index=False))
    print("\n=== 3. MSE = per-target offset + within-target error ===")
    print(t_dec.round(3).to_string(index=False))

    # ---- 4. Ranking vs pK range of the target ----
    rows = []
    for (s, name), pt in t_target.groupby(["set", "predictor"], sort=False):
        pt = pt.dropna(subset=["spearman"])
        q = pd.qcut(pt["pk_range"], 3, labels=["narrow", "middle", "wide"], duplicates="drop")
        for lab, sub in pt.groupby(q, observed=True):
            rows.append({"set": s, "predictor": name, "pk_range_tertile": lab, "n_targets": len(sub),
                         "pk_range_min": sub["pk_range"].min(), "pk_range_max": sub["pk_range"].max(),
                         "mean_spearman": sub["spearman"].mean()})
        rows.append({"set": s, "predictor": name, "pk_range_tertile": "spearman(range, rho)",
                     "n_targets": len(pt), "mean_spearman": stats.spearmanr(pt["pk_range"], pt["spearman"])[0]})
    t_range = pd.DataFrame(rows)
    print("\n=== 4. Ranking vs pK range of the target ===")
    print(t_range.round(3).to_string(index=False))

    # ---- 5. Cluster bootstrap ----
    rows = []
    for s in list(GROUP_SIZE) + list(INDEP_PARENT):
        g = d[d["set"] == s]
        if g.empty:
            continue
        by = {cl: sub for cl, sub in g.groupby("cluster")}
        cl_ids = list(by)
        e_m = {cl: (sub["y_pred"] - sub["y_true"]).to_numpy() for cl, sub in by.items()}
        e_s = {cl: (sub["knn5_pred"] - sub["y_true"]).to_numpy() for cl, sub in by.items()}

        def stat(cs, which):
            em = np.concatenate([e_m[x] for x in cs])
            es = np.concatenate([e_s[x] for x in cs])
            return {"rmse_model": np.sqrt(np.mean(em ** 2)), "rmse_search": np.sqrt(np.mean(es ** 2)),
                    "diff": np.mean(es ** 2 - em ** 2)}[which]

        r = {"set": s, "n": len(g), "n_targets": len(cl_ids)}
        for which in ("rmse_model", "rmse_search", "diff"):
            r[which] = stat(cl_ids, which)
            lo, hi = boot_ci(lambda cs: stat(cs, which), cl_ids, args.n_boot, rng)
            r[f"{which}_ci_low"], r[f"{which}_ci_high"] = lo, hi
        rows.append(r)
    t_boot = pd.DataFrame(rows).rename(columns={"diff": "mean_esearch2_minus_emodel2"})
    print("\n=== 5. Cluster bootstrap (resampling targets) ===")
    print(t_boot.round(3).to_string(index=False))

    for name, t in [("1_ranking_power", t_rank), ("1_per_target", t_target), ("2_pairwise_dpk", t_pairs),
                    ("2b_pairwise_by_tanimoto", t_tan),
                    ("3_offset_vs_within", t_dec), ("4_ranking_vs_range", t_range),
                    ("5_cluster_bootstrap", t_boot)]:
        t.to_csv(os.path.join(out, f"{name}.csv"), index=False)

    # ---- Plot: predicted vs true dpK within targets, CASF-2016 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = d[d["set"] == "casf2016"]
        if len(g):
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
            for ax, (name, col) in zip(axes[:2], PREDICTORS.items()):
                pr = pairs(g, col)
                ax.scatter(pr["dy"], pr["dyhat"], s=8, alpha=0.4, color="#2b6cb0")
                lim = np.abs(pr[["dy", "dyhat"]].to_numpy()).max() * 1.05
                ax.plot([-lim, lim], [-lim, lim], color="#555555", lw=1, ls="--")
                ax.axhline(0, color="#bbbbbb", lw=0.8)
                ax.axvline(0, color="#bbbbbb", lw=0.8)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_xlabel("true dpK within target")
                ax.set_ylabel("predicted dpK")
                ax.set_title(f"CASF-2016, {name}")
            pt = t_target[(t_target["set"] == "casf2016")].dropna(subset=["spearman"])
            for name, color in (("model", "#2b6cb0"), ("search", "#c05621")):
                ax = axes[2]
                ax.hist(pt.loc[pt["predictor"] == name, "spearman"], bins=np.linspace(-1, 1, 11),
                        histtype="step", lw=2, color=color, label=name)
            axes[2].set_xlabel("Spearman within target")
            axes[2].set_ylabel("targets")
            axes[2].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out, "casf2016_ranking.png"), dpi=150)
    except ImportError:
        print("matplotlib not installed, plot skipped")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
