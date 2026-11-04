"""
Within-target ranking on CASF (ranking power) and cluster-level statistics.

CASF-2016 groups its 285 complexes into 57 targets with 5 ligands each and
CASF-2013 its 195 complexes into 65 targets with 3 ligands each. Ranking
power asks whether a model orders the ligands of one target correctly, which
the global Pearson R / RMSE do not show: a model can get the target-level
average right and still rank the ligands of a target poorly, or the reverse.

Clusters. If the official CASF-2016 CoreSet.dat is given (--coreset, column
"target"), it is used for casf2016. Otherwise clusters are rebuilt from the
GEMS protein TM-score matrix: connected components of TM > t among the
complexes of the set, with t chosen on a grid so that as many components as
possible have the official size (5 or 3); average-linkage clustering into the
official number of targets is tried too and the better split is kept. The
script prints the size distribution so you can see how clean it is. Cluster
labels are always computed on the FULL set (all 285 / 195 IDs, from
complexes.csv), so complexes without a prediction (e.g. failed graphs) leave
incomplete groups instead of shifting the clustering; they are listed.

Reads <run_dir>/similarity/similarity_per_complex.csv (from
tools/similarity_analysis.py), which has y_true, the model's y_pred and the
search baseline knn5_pred, and reports for both predictors:
  1. Per-target Spearman and Kendall, averaged over targets (CASF ranking
     power), with a bootstrap 95% CI over targets.
  2. Pairwise differences within a target: error of predicted dpK,
     (yhat_i - yhat_j) - (y_i - y_j), sign accuracy and slope of predicted on
     true dpK, by |true dpK| bin. A target-wide offset cancels here.
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


def cluster_by_tm(ids, idx, M, k):
    pos = np.array([idx[i] for i in ids])
    sub = np.asarray(M[np.ix_(pos, pos)], dtype=np.float64)
    tm = (sub + sub.T) / 2
    np.fill_diagonal(tm, 1.0)
    best = None
    for t in np.round(np.arange(0.50, 0.995, 0.01), 2):
        _, lab = connected_components(csr_matrix(tm > t), directed=False)
        score = n_exact(lab, k)
        if best is None or score > best[0]:
            best = (score, lab, f"connected components, TM > {t:.2f}")
    dist = np.clip(1.0 - tm, 0, None)
    np.fill_diagonal(dist, 0.0)
    lab = fcluster(linkage(squareform(dist, checks=False), "average"),
                   t=len(ids) // k, criterion="maxclust")
    score = n_exact(lab, k)
    if score > best[0]:
        best = (score, lab, f"average linkage into {len(ids) // k} clusters")
    return pd.Series(best[1], index=ids), best[2]


def read_coreset(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            rows.append((parts[0].lower(), int(parts[-1])))
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
        else:
            r["spearman"] = r["kendall"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def pairs(g, col):
    rows = []
    for c, sub in g.groupby("cluster"):
        y, p = sub["y_true"].to_numpy(), sub[col].to_numpy()
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                rows.append({"cluster": c, "dy": y[i] - y[j], "dyhat": p[i] - p[j]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_dir", default=None, help="folder with the GEMS similarity matrices")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--coreset", default=None, help="optional CASF-2016 CoreSet.dat with official targets")
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
    M = idx = None
    for s, k in GROUP_SIZE.items():
        if s == "casf2016" and args.coreset:
            lab = read_coreset(args.coreset)
            how = f"official targets from {os.path.basename(args.coreset)}"
            lab = lab.reindex(full[s])
        else:
            if M is None:
                if args.sim_dir is None:
                    sys.exit("--sim_dir is needed to rebuild the CASF targets from the TM-score matrix")
                with open(find_file(args.sim_dir, "pairwise_similarity_complexes.json")) as f:
                    idx = {pid: n for n, pid in enumerate(json.load(f))}
                M = np.load(find_file(args.sim_dir, "pairwise_similarity_matrix_tm.npy"), mmap_mode="r")
            ids = [i for i in full[s] if i in idx]
            lab, how = cluster_by_tm(ids, idx, M, k)
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
    rank_rows, per_target_all, pair_rows, dec_rows = [], [], [], []
    for s in GROUP_SIZE:
        g = d[d["set"] == s]
        for name, col in PREDICTORS.items():
            pt = per_target(g, col).assign(set=s, predictor=name)
            per_target_all.append(pt)
            ok = pt.dropna(subset=["spearman"]).set_index("cluster")
            lo, hi = boot_ci(lambda cs: ok.loc[cs, "spearman"].mean(), ok.index, args.n_boot, rng)
            rank_rows.append({
                "set": s, "predictor": name, "n_targets": len(ok),
                "mean_spearman": ok["spearman"].mean(), "spearman_ci_low": lo, "spearman_ci_high": hi,
                "mean_kendall": ok["kendall"].mean(),
                "share_targets_spearman_gt_0.5": (ok["spearman"] > 0.5).mean(),
                "global_pearson": np.corrcoef(g[col], g["y_true"])[0, 1],
            })

            # 2. Pairwise dpK
            pr = pairs(g, col)
            pr["eps"] = pr["dyhat"] - pr["dy"]
            pr["bin"] = pd.cut(pr["dy"].abs(), DPK_BINS, labels=DPK_LABELS, right=False)
            nz = pr[pr["dy"] != 0]
            pair_rows.append({"set": s, "predictor": name, "dy_bin": "all", "n_pairs": len(pr),
                              "rmse_dpk": np.sqrt(np.mean(pr["eps"] ** 2)),
                              "sign_accuracy": (np.sign(nz["dyhat"]) == np.sign(nz["dy"])).mean(),
                              "slope_dyhat_on_dy": np.polyfit(pr["dy"], pr["dyhat"], 1)[0]})
            for b, sub in nz.groupby("bin", observed=True):
                pair_rows.append({"set": s, "predictor": name, "dy_bin": b, "n_pairs": len(sub),
                                  "rmse_dpk": np.sqrt(np.mean(sub["eps"] ** 2)),
                                  "sign_accuracy": (np.sign(sub["dyhat"]) == np.sign(sub["dy"])).mean()})

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

    t_rank = pd.DataFrame(rank_rows)
    t_target = pd.concat(per_target_all, ignore_index=True)
    t_pairs = pd.DataFrame(pair_rows)
    t_dec = pd.DataFrame(dec_rows)
    print("\n=== 1. Ranking power: Spearman / Kendall within targets, averaged over targets ===")
    print(t_rank.round(3).to_string(index=False))
    print("\n=== 2. Pairwise dpK within targets: (yhat_i - yhat_j) - (y_i - y_j) ===")
    print(t_pairs.round(3).to_string(index=False))
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
