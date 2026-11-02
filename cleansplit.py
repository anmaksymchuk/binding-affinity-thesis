"""
PDBbind v2020.R1 + CleanSplit: master table of complexes and split logic.

Run once after unpacking the R1 structures (and every time the data,
the index file or the label filters change):

    python cleansplit.py

It writes `config.complexes_csv` (one row per PDB ID from the CleanSplit
JSON) and prints a coverage report. `main.py` and `dataset.py` read this
table, so it is the single source of IDs, labels and file paths.

Columns of complexes.csv
------------------------
pdb_id               lowercase PDB ID
in_train             in CleanSplit "train"
in_casf2016, in_casf2016_indep, in_casf2013, in_casf2013_indep
val_fold             k if the ID is in the "validation" list of GEMS fold k,
                     -1 if it is in none of the fold files
pk                   -log10(Kd/Ki/IC50 in M) from the chosen label source
label_type           Kd / Ki / IC50
precision            = < <= > >= ~
measurement          raw string, e.g. "Kd=0.43uM"
label_source         r1_index or gems_dict
pk_v2020             label from the original v2020 (GEMS data dict), for comparison
resolution, year
has_structure        ligand and pocket files were found under config.data_dir
ligand_file, pocket_file   paths relative to config.data_dir
exclude_reason       "" if usable, otherwise no_structure / no_label /
                     censored_label / approx_label
                     (graph-building failures are added later by dataset.py)
"""

import json
import math
import os
import re
import hashlib
from glob import glob

import numpy as np
import pandas as pd


SPLIT_KEYS = ["train", "casf2016", "casf2016_indep", "casf2013", "casf2013_indep"]
TEST_KEYS = ["casf2016", "casf2016_indep", "casf2013", "casf2013_indep"]

PDB_ID_RE = re.compile(r"^[0-9][a-z0-9]{3}$")

MEASUREMENT_RE = re.compile(
    r"\b(Kd|Ki|IC50)\s*(<=|>=|=|<|>|~)\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(fM|pM|nM|uM|mM|M)\b"
)

UNIT_TO_MOLAR = {
    "fM": 1e-15,
    "pM": 1e-12,
    "nM": 1e-9,
    "uM": 1e-6,
    "mM": 1e-3,
    "M": 1.0,
}

CENSORED = {"<", "<=", ">", ">="}
APPROX = {"~"}


# ============================================================
# Small helpers
# ============================================================
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fold_file(config, fold):
    return os.path.join(config.split_dir, config.fold_file_pattern.format(fold=fold))


def label_exclusion_reason(precision, config):
    """
    Single place where label filters are applied (used by prepare and main).
    """
    if precision in CENSORED and config.drop_censored_labels:
        return "censored_label"
    if precision in APPROX and not config.keep_approx_labels:
        return "approx_label"
    return ""


# ============================================================
# Inputs
# ============================================================
def load_cleansplit(config):
    with open(config.split_file) as f:
        split = json.load(f)

    split = {k: [str(x).lower() for x in v] for k, v in split.items()}

    missing_keys = [k for k in SPLIT_KEYS if k not in split]
    if missing_keys:
        raise KeyError(f"Split file {config.split_file} has no keys {missing_keys}")

    return split


def load_gems_folds(config, n_folds=5):
    """
    Returns {pdb_id: k} for IDs in the "validation" list of fold k.
    """
    id_to_fold = {}

    for k in range(n_folds):
        path = fold_file(config, k)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fold file not found: {path}")

        with open(path) as f:
            fold = json.load(f)

        for pdb_id in fold["validation"]:
            pdb_id = pdb_id.lower()
            if pdb_id in id_to_fold:
                raise RuntimeError(f"{pdb_id} is in the validation set of two folds")
            id_to_fold[pdb_id] = k

    return id_to_fold


def parse_measurement(text):
    """
    "Kd=0.43uM" -> ("Kd", "=", 6.3665, "Kd=0.43uM"). Returns None if not found.
    """
    m = MEASUREMENT_RE.search(text)
    if m is None:
        return None

    label_type, precision, value, unit = m.groups()
    molar = float(value) * UNIT_TO_MOLAR[unit]

    if molar <= 0:
        return None

    pk = -math.log10(molar)
    return label_type, precision, pk, m.group(0).replace(" ", "")


def find_r1_index(config):
    """
    Use config.r1_index_file if it exists; otherwise look for a PDBbind
    general index file under data_dir (the exact name differs between releases).
    """
    if config.r1_index_file and os.path.exists(config.r1_index_file):
        return config.r1_index_file

    patterns = ["**/INDEX_general_PL*", "**/INDEX_*general*", "**/index/*.lst"]
    candidates = []
    for pattern in patterns:
        candidates += glob(os.path.join(config.data_dir, pattern), recursive=True)
        candidates += glob(os.path.join(config.project_dir, pattern), recursive=True)

    candidates = sorted({c for c in candidates if os.path.isfile(c)})

    if len(candidates) == 1:
        print(f"Found R1 index file: {candidates[0]}")
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"R1 index file not found at {config.r1_index_file} and no INDEX_general_PL* "
            f"file under {config.data_dir}. Set config.r1_index_file to the index file "
            "from the PDBbind v2020.R1 download, or set label_source = 'gems_dict'."
        )

    raise FileNotFoundError(
        "Several candidate index files found, set config.r1_index_file explicitly:\n  "
        + "\n  ".join(candidates)
    )


def parse_r1_index(path):
    """
    Parse a PDBbind index file (general/refined, any release).

    Expected line layout (as in PDBbind INDEX_general_PL_data.20xx):
        PDB_code  resolution  release_year  -logKd/Ki  Kd/Ki/IC50  // reference (ligand)

    The label type, precision and value are read from the measurement string,
    so the parser does not depend on the exact column order. If the 4th column
    is a -logK value that agrees with the measurement (within 0.01), that
    published value is used; otherwise -log10(measurement) is used and the
    line is counted as a mismatch.
    """
    labels = {}
    n_unparsed = 0
    n_mismatch = 0

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            pdb_id = parts[0].lower()
            if not PDB_ID_RE.match(pdb_id):
                continue

            parsed = parse_measurement(line)
            if parsed is None:
                n_unparsed += 1
                continue

            label_type, precision, pk, measurement = parsed

            resolution = parts[1] if len(parts) > 1 else ""
            year = parts[2] if len(parts) > 2 and parts[2].isdigit() else ""

            # Prefer the published -logK column (rounded to 2 decimals, as in
            # PDBbind and GEMS) when it agrees with the measurement string.
            if len(parts) > 3:
                try:
                    pk_col = float(parts[3])
                except ValueError:
                    pk_col = None

                if pk_col is not None:
                    if abs(pk_col - pk) <= 0.01:
                        pk = pk_col
                    else:
                        n_mismatch += 1

            labels[pdb_id] = {
                "pk": pk,
                "label_type": label_type,
                "precision": precision,
                "measurement": measurement,
                "resolution": resolution,
                "year": year,
            }

    print(f"Parsed {len(labels)} labels from {path}")
    if n_unparsed:
        print(f"  lines without Kd/Ki/IC50 measurement: {n_unparsed}")
    if n_mismatch:
        print(
            f"  Warning: for {n_mismatch} lines the -logK column differs from "
            "-log10(measurement) by > 0.01; the value from the measurement is used."
        )

    return labels


def load_gems_labels(path):
    with open(path) as f:
        d = json.load(f)

    labels = {}
    for pdb_id, rec in d.items():
        label_type = next((t for t in ("Kd", "Ki", "IC50") if t in rec), "")
        labels[pdb_id.lower()] = {
            "pk": float(rec["log_kd_ki"]),
            "label_type": label_type,
            "precision": rec.get("precision", ""),
            "measurement": "",
            "resolution": rec.get("resolution", ""),
            "year": "",
        }
    return labels


def index_structures(data_dir, wanted_ids):
    """
    Walk data_dir (any nesting) and find <id>_ligand.mol2/.sdf and <id>_pocket.pdb.
    Returns {pdb_id: {"ligand_file": rel_path, "pocket_file": rel_path}}.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Structure directory not found: {data_dir}. Unpack the PDBbind v2020.R1 "
            "'Protein-ligand complex structures' archive there or change config.data_dir."
        )

    found = {}

    for dirpath, _, filenames in os.walk(data_dir):
        names = set(filenames)
        for name in filenames:
            if not name.endswith("_pocket.pdb"):
                continue

            pdb_id = name[: -len("_pocket.pdb")].lower()
            if pdb_id not in wanted_ids:
                continue

            ligand = None
            for ext in ("mol2", "sdf"):
                cand = f"{name[: -len('_pocket.pdb')]}_ligand.{ext}"
                if cand in names:
                    ligand = cand
                    break

            if ligand is None:
                continue

            if pdb_id in found:
                print(f"  Warning: {pdb_id} found twice, keeping {found[pdb_id]['pocket_file']}")
                continue

            found[pdb_id] = {
                "ligand_file": os.path.relpath(os.path.join(dirpath, ligand), data_dir),
                "pocket_file": os.path.relpath(os.path.join(dirpath, name), data_dir),
            }

    return found


# ============================================================
# Master table
# ============================================================
def build_complexes_table(config):
    split = load_cleansplit(config)
    id_to_fold = load_gems_folds(config)

    all_ids = sorted(set().union(*[set(split[k]) for k in SPLIT_KEYS]))
    sets = {k: set(split[k]) for k in SPLIT_KEYS}

    gems_labels = load_gems_labels(config.gems_data_dict)

    if config.label_source == "r1_index":
        labels = parse_r1_index(find_r1_index(config))
    elif config.label_source == "gems_dict":
        labels = gems_labels
    else:
        raise ValueError(f"Unknown label_source: {config.label_source}")

    print(f"Indexing structures under {config.data_dir} ...")
    structures = index_structures(config.data_dir, set(all_ids))

    rows = []
    for pdb_id in all_ids:
        lab = labels.get(pdb_id)
        st = structures.get(pdb_id)
        gems = gems_labels.get(pdb_id)

        if st is None:
            reason = "no_structure"
        elif lab is None:
            reason = "no_label"
        else:
            reason = label_exclusion_reason(lab["precision"], config)

        rows.append(
            {
                "pdb_id": pdb_id,
                **{f"in_{k}": pdb_id in sets[k] for k in SPLIT_KEYS},
                "val_fold": id_to_fold.get(pdb_id, -1),
                "pk": lab["pk"] if lab else np.nan,
                "label_type": lab["label_type"] if lab else "",
                "precision": lab["precision"] if lab else "",
                "measurement": lab["measurement"] if lab else "",
                "label_source": config.label_source if lab else "",
                "pk_v2020": gems["pk"] if gems else np.nan,
                "resolution": lab["resolution"] if lab else (gems["resolution"] if gems else ""),
                "year": lab["year"] if lab else "",
                "has_structure": st is not None,
                "ligand_file": st["ligand_file"] if st else "",
                "pocket_file": st["pocket_file"] if st else "",
                "exclude_reason": reason,
            }
        )

    return pd.DataFrame(rows)


def load_complexes_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python cleansplit.py` first to build it."
        )

    table = pd.read_csv(
        path,
        dtype={"pdb_id": str, "precision": str, "measurement": str,
               "label_type": str, "ligand_file": str, "pocket_file": str,
               "exclude_reason": str, "resolution": str, "year": str},
        keep_default_na=False,
        na_values={"pk": [""], "pk_v2020": [""]},
    )
    return table


def coverage_report(table):
    lines = []
    lines.append(f"{'set':<16}{'IDs':>7}{'struct':>8}{'usable':>8}   excluded")

    for k in SPLIT_KEYS:
        sub = table[table[f"in_{k}"]]
        usable = (sub["exclude_reason"] == "").sum()
        reasons = sub.loc[sub["exclude_reason"] != "", "exclude_reason"].value_counts().to_dict()
        lines.append(
            f"{k:<16}{len(sub):>7}{int(sub['has_structure'].sum()):>8}{usable:>8}   {reasons}"
        )

    both = table[table["pk"].notna() & table["pk_v2020"].notna()]
    changed = both[(both["pk"] - both["pk_v2020"]).abs() > 0.01]
    lines.append(f"labels differing from v2020 by > 0.01 pK: {len(changed)}")
    if 0 < len(changed) <= 20:
        for _, r in changed.iterrows():
            lines.append(f"  {r.pdb_id}: v2020 {r.pk_v2020:.2f} -> {r.pk:.2f} ({r.measurement})")

    return "\n".join(lines)


# ============================================================
# Split used by main.py
# ============================================================
def make_split(table, available_ids, config, skipped=None, rng=None):
    """
    Build train / val / test ID lists.

    table          complexes.csv as DataFrame
    available_ids  IDs that have a processed graph
    skipped        {pdb_id: reason} from dataset.py for graphs that failed
    """
    skipped = skipped or {}
    available_ids = set(available_ids)

    t = table.copy()
    # Data reasons come from the table; label filters are re-applied with the
    # current config, so changing them does not require re-running this script.
    data_reason = t["exclude_reason"].where(t["exclude_reason"].isin(["no_structure", "no_label"]), "")
    t["label_reason"] = [label_exclusion_reason(p, config) for p in t["precision"]]
    t["final_reason"] = data_reason.where(data_reason != "", t["label_reason"])
    no_graph = (t["final_reason"] == "") & ~t["pdb_id"].isin(available_ids)
    t.loc[no_graph, "final_reason"] = [
        f"graph_failed:{skipped.get(i, 'not_processed')}" for i in t.loc[no_graph, "pdb_id"]
    ]

    usable = t["final_reason"] == ""

    excluded = {
        k: {r.pdb_id: r.final_reason for r in t[t[f"in_{k}"] & ~usable].itertuples()}
        for k in SPLIT_KEYS
    }

    train_pool = t[t["in_train"] & usable]

    if config.val_mode == "gems_fold":
        val_mask = train_pool["val_fold"] == config.fold
        val_ids = sorted(train_pool.loc[val_mask, "pdb_id"])
        train_ids = sorted(train_pool.loc[~val_mask, "pdb_id"])
    elif config.val_mode == "random":
        rng = rng or np.random.default_rng(config.seed)
        ids = np.array(sorted(train_pool["pdb_id"]))
        rng.shuffle(ids)
        n_val = int(len(ids) * config.val_split)
        if n_val == 0:
            raise ValueError("Validation split is empty.")
        val_ids = sorted(ids[:n_val].tolist())
        train_ids = sorted(ids[n_val:].tolist())
    else:
        raise ValueError(f"Unknown val_mode: {config.val_mode}. Use 'gems_fold' or 'random'.")

    tests = {
        k: sorted(t.loc[t[f"in_{k}"] & usable, "pdb_id"])
        for k in config.test_sets
    }

    # ---------------------------
    # Leakage checks (by ID)
    # ---------------------------
    all_test = set().union(*[set(t.loc[t[f"in_{k}"], "pdb_id"]) for k in TEST_KEYS])

    if set(train_ids) & set(val_ids):
        raise RuntimeError("Train/Val overlap")
    for name, ids in (("Train", train_ids), ("Val", val_ids)):
        overlap = set(ids) & all_test
        if overlap:
            raise RuntimeError(f"{name} contains CASF IDs: {sorted(overlap)[:10]}")

    if not train_ids or not val_ids:
        raise RuntimeError(f"Empty split: train={len(train_ids)}, val={len(val_ids)}")

    counts = {
        k: {
            "in_split_file": int(t[f"in_{k}"].sum()),
            "used": int((t[f"in_{k}"] & usable).sum()),
            "excluded_by_reason": t.loc[t[f"in_{k}"] & ~usable, "final_reason"].value_counts().to_dict(),
        }
        for k in SPLIT_KEYS
    }

    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": tests,
        "excluded": excluded,
        "counts": counts,
    }


# ============================================================
# CLI
# ============================================================
def main():
    from config import Config

    config = Config()

    table = build_complexes_table(config)

    os.makedirs(os.path.dirname(config.complexes_csv), exist_ok=True)
    table.to_csv(config.complexes_csv, index=False)

    print("\n--- CleanSplit coverage ---")
    print(f"label_source = {config.label_source}, drop_censored_labels = "
          f"{config.drop_censored_labels}, keep_approx_labels = {config.keep_approx_labels}")
    print(coverage_report(table))
    print(f"\nSaved {len(table)} rows to {config.complexes_csv}")


if __name__ == "__main__":
    main()
