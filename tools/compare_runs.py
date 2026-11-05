"""
Per-complex comparison of two runs that share the same split (e.g. full graph
with all pocket hydrogens vs. the fast config without them).

For every complex of val and of the CASF sets it takes the error of run A and
of run B and asks where B is better or worse, and whether that depends on
properties of the complex that the graph representation touches:
    pocket_atoms, pocket_h, pocket_h_frac   from <id>_pocket.pdb
    ligand_heavy_atoms                      from <id>_ligand.mol2 / .sdf
    pk                                      true label
    max_tanimoto, nn_dpk                    from <run_a>/similarity/similarity_per_complex.csv,
                                            if present
Reported per group (val, casf2016, all CASF complexes pooled):
  1. RMSE of A and B and the paired bootstrap CI of RMSE_B - RMSE_A
     (over complexes; for CASF also over targets if ranking/casf_clusters.csv exists);
     correlation of the two runs' errors; share of complexes where B is better.
  2. For each property: quartile bins with the mean of e_B^2 - e_A^2 (negative =
     B better) and its bootstrap CI, and the Spearman correlation of the property
     with |e_B| - |e_A|.

Two single runs also differ by training noise (seed, order of batches), so a
per-complex difference is meaningful only as a consistent trend over many
complexes; val (2427 complexes) has far more power than CASF-2016 (283).

Usage (from the repo root):
    python tools/compare_runs.py --run_a outputs/gnn_only_val15_seed42 \
        --run_b outputs/gnn_only_noH_amp_val15_seed42 --labels allH noH_amp

Writes <run_b>/compare_vs_<label A>/: per_complex.csv, overall.csv, by_property.csv, by_property.png
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from scipy import stats

CASF = ["casf2016", "casf2016_indep", "casf2013", "casf2013_indep"]


def load_preds(run_dir):
    frames = []
    for f in ("val_predictions.csv", "test_predictions.csv"):
        p = os.path.join(run_dir, f)
        if not os.path.exists(p):
            sys.exit(f"{p} not found (run tools/evaluate.py on {run_dir} first)")
        frames.append(pd.read_csv(p, dtype={"pdb_id": str}))
    return pd.concat(frames).rename(columns={"test_set": "set"})[["set", "pdb_id", "y_true", "y_pred"]]


def pocket_counts(path):
    n = nh = 0
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    if line[17:20] == "HOH":
                        continue
                    el = line[76:78].strip() or line[12:16].strip().lstrip("0123456789")[:1]
                    n += 1
                    nh += el.upper() in ("H", "D")
    except OSError:
        return np.nan, np.nan
    return n, nh


def ligand_heavy_atoms(path):
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return np.nan
    if path.endswith(".mol2"):
        block = text.split("@<TRIPOS>ATOM")[1].split("@<TRIPOS>")[0]
        types = [ln.split()[5] for ln in block.strip().splitlines() if len(ln.split()) >= 6]
        return sum(not t.upper().startswith("H") for t in types)
    lines = text.splitlines()
    n_atoms = int(lines[3][:3])
    return sum(lines[4 + i].split()[3].upper() not in ("H", "D") for i in range(n_atoms))


def boot_ci(fn, units, n_boot, rng):
    units = np.asarray(units)
    vals = [fn(rng.choice(units, len(units), replace=True)) for _ in range(n_boot)]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", required=True)
    ap.add_argument("--run_b", required=True)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    la, lb = args.labels
    rng = np.random.default_rng(args.seed)
    out = os.path.join(args.run_b, f"compare_vs_{la}")
    os.makedirs(out, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    a, b = load_preds(args.run_a), load_preds(args.run_b)
    d = a.merge(b, on=["set", "pdb_id"], suffixes=("_a", "_b"))
    if not np.allclose(d["y_true_a"], d["y_true_b"]):
        sys.exit("y_true differs between the runs: they do not use the same labels")
    print(f"Matched complexes: {len(d)} (A {len(a)}, B {len(b)})")
    d = d.rename(columns={"y_true_a": "pk"}).drop(columns="y_true_b")
    d["e_a"], d["e_b"] = d["y_pred_a"] - d["pk"], d["y_pred_b"] - d["pk"]
    d["d_abs"] = d["e_b"].abs() - d["e_a"].abs()
    d["d_sq"] = d["e_b"] ** 2 - d["e_a"] ** 2

    # ---- properties of the complexes ----
    from config import Config
    c = Config()
    if os.path.exists(c.complexes_csv):
        t = pd.read_csv(c.complexes_csv, dtype={"pdb_id": str}).set_index("pdb_id")
        ids = d["pdb_id"].unique()
        print(f"Reading {len(ids)} pocket / ligand files ...", flush=True)
        props = {}
        for pid in ids:
            if pid not in t.index or not isinstance(t.at[pid, "pocket_file"], str):
                continue
            n, nh = pocket_counts(os.path.join(c.data_dir, t.at[pid, "pocket_file"]))
            props[pid] = {"pocket_atoms": n, "pocket_h": nh,
                          "pocket_h_frac": nh / n if n else np.nan,
                          "ligand_heavy_atoms": ligand_heavy_atoms(os.path.join(c.data_dir, t.at[pid, "ligand_file"]))}
        d = d.merge(pd.DataFrame.from_dict(props, orient="index").rename_axis("pdb_id").reset_index(),
                    on="pdb_id", how="left")
    else:
        print(f"Note: {c.complexes_csv} not found, structure properties skipped")
    simf = os.path.join(args.run_a, "similarity", "similarity_per_complex.csv")
    if os.path.exists(simf):
        s = pd.read_csv(simf, dtype={"pdb_id": str})[["set", "pdb_id", "max_tanimoto", "nn_pk"]]
        d = d.merge(s, on=["set", "pdb_id"], how="left")
        d["nn_dpk"] = (d["pk"] - d["nn_pk"]).abs()
        d = d.drop(columns="nn_pk")
    clusters = None
    for run in (args.run_a, args.run_b):
        cf = os.path.join(run, "similarity", "ranking", "casf_clusters.csv")
        if os.path.exists(cf):
            cc = pd.read_csv(cf, dtype={"pdb_id": str, "cluster": str})
            clusters = {(r.set, r.pdb_id): r.cluster for r in cc.itertuples()}
            break
    d.to_csv(os.path.join(out, "per_complex.csv"), index=False)

    groups = {"val": d[d["set"] == "val"], "casf2016": d[d["set"] == "casf2016"],
              "casf_all (unique)": d[d["set"].isin(CASF)].drop_duplicates("pdb_id")}

    # ---- 1. overall ----
    rows = []
    for gname, g in groups.items():
        if g.empty:
            continue
        ea, eb = g["e_a"].to_numpy(), g["e_b"].to_numpy()
        idx = np.arange(len(g))
        diff = lambda ii: np.sqrt(np.mean(eb[ii] ** 2)) - np.sqrt(np.mean(ea[ii] ** 2))
        lo, hi = boot_ci(diff, idx, args.n_boot, rng)
        r = {"group": gname, "n": len(g), f"rmse_{la}": np.sqrt(np.mean(ea ** 2)),
             f"rmse_{lb}": np.sqrt(np.mean(eb ** 2)), "rmse_diff_b_minus_a": diff(idx),
             "ci_low": lo, "ci_high": hi, "corr_errors": np.corrcoef(ea, eb)[0, 1],
             f"share_{lb}_better": (g["d_abs"] < 0).mean()}
        if clusters is not None and gname == "casf2016":
            cl = np.array([clusters.get(("casf2016", p)) for p in g["pdb_id"]])
            ok = cl != None  # noqa: E711
            units = {u: np.where(cl == u)[0] for u in set(cl[ok])}
            lo, hi = boot_ci(lambda us: diff(np.concatenate([units[u] for u in us])), list(units), args.n_boot, rng)
            r["ci_low_by_target"], r["ci_high_by_target"] = lo, hi
        rows.append(r)
    t_all = pd.DataFrame(rows)
    print(f"\n=== 1. {lb} vs {la}: RMSE and paired bootstrap of RMSE_{lb} - RMSE_{la} (negative = {lb} better) ===")
    print(t_all.round(3).to_string(index=False))

    # ---- 2. by property ----
    prop_cols = [p for p in ["pocket_atoms", "pocket_h", "pocket_h_frac", "ligand_heavy_atoms", "pk",
                             "max_tanimoto", "nn_dpk"] if p in d.columns and d[p].notna().any()]
    rows = []
    for gname in ("val", "casf2016"):
        g = groups[gname]
        for p in prop_cols:
            gg = g.dropna(subset=[p])
            if len(gg) < 20:
                continue
            q = pd.qcut(gg[p], 4, duplicates="drop")
            for k, (lab, sub) in enumerate(gg.groupby(q, observed=True)):
                x = sub["d_sq"].to_numpy()
                lo, hi = boot_ci(lambda ii: x[ii].mean(), np.arange(len(x)), args.n_boot, rng)
                rows.append({"group": gname, "property": p, "quartile": f"Q{k + 1}",
                             "range": f"{sub[p].min():.3g}-{sub[p].max():.3g}", "n": len(sub),
                             "mean_d_sq": x.mean(), "ci_low": lo, "ci_high": hi,
                             f"share_{lb}_better": (sub["d_abs"] < 0).mean()})
            rho, pv = stats.spearmanr(gg[p], gg["d_abs"])
            rows.append({"group": gname, "property": p, "quartile": "spearman(prop, d_abs)",
                         "n": len(gg), "mean_d_sq": rho, "ci_low": pv, "ci_high": np.nan})
    t_prop = pd.DataFrame(rows)
    print(f"\n=== 2. e_{lb}^2 - e_{la}^2 by property quartile (negative = {lb} better); "
          f"last row per property: Spearman rho and p-value ===")
    print(t_prop.round(3).to_string(index=False))

    t_all.to_csv(os.path.join(out, "overall.csv"), index=False)
    t_prop.to_csv(os.path.join(out, "by_property.csv"), index=False)

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        q = t_prop[t_prop["quartile"].str.startswith("Q")]
        props = list(dict.fromkeys(q["property"]))
        if props:
            fig, axes = plt.subplots(1, len(props), figsize=(3.2 * len(props), 3.6), sharey=True, squeeze=False)
            for ax, p in zip(axes[0], props):
                for gname, color, dx in (("val", "#2a78d6", -0.12), ("casf2016", "#eb6834", 0.12)):
                    sub = q[(q["property"] == p) & (q["group"] == gname)]
                    xs = np.arange(len(sub)) + dx
                    ax.errorbar(xs, sub["mean_d_sq"], yerr=[sub["mean_d_sq"] - sub["ci_low"], sub["ci_high"] - sub["mean_d_sq"]],
                                fmt="o", ms=5, color=color, capsize=2, lw=1, label=gname)
                ax.axhline(0, color="#52514e", lw=0.8)
                ax.set_xticks(range(4))
                ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=8)
                ax.set_title(p, fontsize=10)
                ax.grid(axis="y", color="#e4e3df", lw=0.6)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
            axes[0][0].set_ylabel(f"mean e²({lb}) − e²({la}), pK²\n(below 0: {lb} better)")
            axes[0][0].legend(fontsize=8, frameon=False)
            fig.suptitle(f"{lb} vs {la} by property quartile (95% bootstrap CI)", fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "by_property.png"), dpi=170)
    except ImportError:
        print("matplotlib not installed, plot skipped")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
