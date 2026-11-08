"""
Check on real graphs that graph_input = "ligand" really leaves the ligand alone.

Goes through the DataLoader, the same path training uses, so it also proves
that the transform is actually applied to the batches and not merely defined.
For every sampled complex it compares the ligand view against the untransformed
complex graph (dataset.get bypasses the transform) and asserts:

    only ligand atoms are left        x[:, 43] > 0.5 everywhere, count matches
    atom features are untouched       kept rows equal the complex graph's rows
    both branches carry the same      edge_index_1 == edge_index_2
    no contact edges are left         edge_type == 0, edge_weight == 1.0
    ligand bonds are all kept         count equals the complex graph's
    dropped edges all touched pocket  no ligand-ligand edge was lost
    indices stay in range             max index < num_nodes, no NaN features

Run it once before a ligand-only training run, with the config of that run:
the graph cache it reads follows pocket_hydrogens, so set the pilot's values
first and only then check.

    python tools/check_ligand_only.py [--n 500] [--batch_size 32] [--seed 42]

With --compare_split <run_dir> it also recomputes the split from the current
config and checks that train, val and every test set hold exactly the same
complexes as that finished run, so the pilot is comparable per complex.

Exits with a non-zero status if any check fails.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from config import Config
from dataset import Ligand_Protein_Dataset
from cleansplit import load_complexes_table, make_split
from utils import ligand_only_view, LIGAND_INDICATOR_IDX


def sort_pairs(edge_index):
    """Edges as a sorted list of (u, v), so two edge lists compare directly."""
    if edge_index.numel() == 0:
        return edge_index.reshape(2, 0)

    order = np.lexsort((edge_index[1].numpy(), edge_index[0].numpy()))

    return edge_index[:, torch.from_numpy(order.copy())]


def check_batch(batch):
    """Invariants that must hold for a batch of ligand-only graphs."""
    assert torch.all(batch.x[:, LIGAND_INDICATOR_IDX] > 0.5), "pocket atom in batch"
    assert torch.isfinite(batch.x).all(), "non-finite atom feature"

    assert batch.edge_index_1.shape == batch.edge_index_2.shape, "branch shapes differ"
    assert torch.equal(batch.edge_index_1, batch.edge_index_2), "branches differ"

    assert torch.all(batch.edge_type == 0), "contact edge left in batch"
    assert torch.all(batch.edge_weight == 1.0), "non-covalent edge weight left"
    assert batch.edge_weight.shape[0] == batch.edge_index_2.shape[1], "edge_weight length"
    assert batch.edge_type.shape[0] == batch.edge_index_2.shape[1], "edge_type length"

    n = batch.num_nodes
    for name in ("edge_index_1", "edge_index_2"):
        ei = getattr(batch, name)
        if ei.numel():
            assert int(ei.min()) >= 0 and int(ei.max()) < n, f"{name} out of range"


def check_against_complex(lig, full):
    """Compare one ligand view with the untransformed complex graph."""
    pdb_id = full.pdb_id[0] if isinstance(full.pdb_id, list) else full.pdb_id
    keep = full.x[:, LIGAND_INDICATOR_IDX] > 0.5
    n_lig = int(keep.sum())

    assert n_lig > 0, f"{pdb_id}: complex graph has no ligand atoms"
    assert lig.num_nodes == n_lig, f"{pdb_id}: {lig.num_nodes} nodes, expected {n_lig}"
    assert torch.equal(lig.x, full.x[keep]), f"{pdb_id}: atom features changed"
    assert torch.equal(lig.y, full.y), f"{pdb_id}: label changed"

    # The edge lists must match pair for pair, not merely in count: the
    # ligand-internal edges of the complex graph, renumbered the same way,
    # have to be exactly the edges of the ligand view.
    remap = torch.full((full.x.size(0),), -1, dtype=torch.long)
    remap[keep] = torch.arange(n_lig, dtype=torch.long)

    for name in ("edge_index_1", "edge_index_2"):
        ei_full = getattr(full, name)
        inside = keep[ei_full[0]] & keep[ei_full[1]]
        expected = remap[ei_full[:, inside]]
        got = getattr(lig, name)

        assert got.shape[1] == int(inside.sum()), (
            f"{pdb_id}: {name} kept {got.shape[1]}, ligand-internal {int(inside.sum())}")
        assert torch.equal(sort_pairs(got), sort_pairs(expected)), (
            f"{pdb_id}: {name} holds different edges than the ligand bonds")

    # Every edge of the complex graph that was dropped must touch a pocket atom.
    ei_full = full.edge_index_2
    dropped = ~(keep[ei_full[0]] & keep[ei_full[1]])
    if int(dropped.sum()):
        ends = torch.cat([ei_full[0][dropped], ei_full[1][dropped]])
        assert not bool(keep[ends].all()), f"{pdb_id}: ligand-ligand edge dropped"

    return {"pdb_id": pdb_id, "nodes": n_lig, "edges": lig.edge_index_2.shape[1],
            "nodes_complex": int(full.x.shape[0])}


def compare_split(ds, c, ref_run_dir):
    """
    Check that the split the current config produces is the one a finished run
    used, so the ligand-only pilot trains and tests on the same complexes as
    the full model it will be compared with.
    """
    with open(os.path.join(ref_run_dir, "split_info.json")) as f:
        ref = json.load(f)

    table = load_complexes_table(c.complexes_csv)
    ids = [d.pdb_id[0] if isinstance(d.pdb_id, list) else d.pdb_id
           for d in (ds.get(i) for i in range(len(ds)))]
    split = make_split(table, {str(i).lower() for i in ids}, c, skipped=ds.skipped)

    print(f"\nSplit against {os.path.basename(os.path.normpath(ref_run_dir))}"
          f" (its val_mode {ref['val_mode']}, fold {ref['fold']},"
          f" graph_input {ref.get('graph_input', 'complex')}):")

    same = True
    pairs = [("train", split["train_ids"], ref["train_ids"]),
             ("val", split["val_ids"], ref["val_ids"])]
    pairs += [(name, split["test_ids"][name], ref["test_ids"][name])
              for name in split["test_ids"] if name in ref["test_ids"]]

    for name, mine, theirs in pairs:
        equal = set(mine) == set(theirs)
        same = same and equal
        print(f"  {name:<16}{len(mine):>6} vs {len(theirs):<6} "
              f"{'identical' if equal else 'DIFFERENT'}")

    if not same:
        raise SystemExit(
            "\nThe split differs from the reference run. Check val_mode, fold, "
            "seed, the label filters and the graph cache before training, "
            "otherwise the two models are not comparable per complex.")

    print("  Same train, val and test complexes as the reference run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="complexes to check")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare_split", default=None, metavar="RUN_DIR",
                    help="also check that the split of the current config matches "
                         "this finished run's split_info.json")
    args = ap.parse_args()

    c = Config()

    # The graph cache and the split follow config.py, so run this script with
    # the settings of the run it is checking, not with the defaults.
    print(f"pocket_hydrogens: {c.pocket_hydrogens}")
    print(f"graph_input:      {getattr(c, 'graph_input', 'complex')}")
    print(f"val_mode:         {c.val_mode}"
          + (f", fold {c.fold}" if c.val_mode == "gems_fold" else ""))

    if getattr(c, "graph_input", "complex") != "ligand":
        print("\nNote: config.graph_input is not \"ligand\". This script checks the\n"
              "ligand view anyway, but `python main.py` with this config would train\n"
              "on the complexes. Set graph_input = \"ligand\" before the pilot, and\n"
              "check the cache the pilot will use (pocket_hydrogens).")

    # The dataset carries the transform, exactly as main.py sets it up with
    # graph_input = "ligand", whatever config.py currently says.
    ds = Ligand_Protein_Dataset(
        transform=ligand_only_view,
        root=c.root,
        data_dir=c.data_dir,
        complexes_csv=c.complexes_csv,
        dataset_tag=c.dataset_tag,
        shard_size=c.graph_shard_size,
        pocket_hydrogens=c.pocket_hydrogens,
        esm_path=c.esm_path,
        esm_dim=c.esm_dim,
    )

    n_total = len(ds)
    if n_total == 0:
        raise RuntimeError("Graph cache is empty for this dataset_tag.")

    n = min(args.n, n_total)
    idx = np.random.default_rng(args.seed).choice(n_total, size=n, replace=False)
    idx = sorted(int(i) for i in idx)

    print(f"Graph cache:  {c.dataset_tag}")
    print(f"Checking {n} of {n_total} complexes\n")

    rows = []
    for i in idx:
        # get() returns the stored complex graph (no transform), ds[i] the view.
        # The complex graph is cloned first, so the comparison cannot be fooled
        # by a PyG version that hands out the same object twice.
        full = ds.get(i).clone()
        rows.append(check_against_complex(ds[i], full))

    # The DataLoader path, the one training uses.
    n_batches, n_graphs = 0, 0
    for batch in DataLoader(ds[idx], batch_size=args.batch_size, shuffle=False):
        check_batch(batch)
        n_batches += 1
        n_graphs += int(batch.batch.max()) + 1

    assert n_graphs == n, f"DataLoader returned {n_graphs} graphs, expected {n}"

    nodes = np.array([r["nodes"] for r in rows])
    edges = np.array([r["edges"] for r in rows])
    full_nodes = np.array([r["nodes_complex"] for r in rows])
    empty = int((edges == 0).sum())

    print("All checks passed.\n")
    print(f"{'':<22}{'min':>8}{'median':>9}{'max':>8}{'mean':>9}")
    print(f"{'ligand atoms':<22}{nodes.min():>8}{int(np.median(nodes)):>9}"
          f"{nodes.max():>8}{nodes.mean():>9.1f}")
    print(f"{'ligand bond ends':<22}{edges.min():>8}{int(np.median(edges)):>9}"
          f"{edges.max():>8}{edges.mean():>9.1f}")
    print(f"{'atoms in complex':<22}{full_nodes.min():>8}{int(np.median(full_nodes)):>9}"
          f"{full_nodes.max():>8}{full_nodes.mean():>9.1f}")
    print(f"\nGraph is {nodes.sum() / full_nodes.sum():.1%} of the complex by atoms "
          f"({n_batches} batches checked through the DataLoader)")

    if empty:
        print(f"Warning: {empty} of {n} ligands have no bonds in the cache "
              f"(single atoms or a parsing problem)")

    if args.compare_split:
        compare_split(ds, c, args.compare_split)


if __name__ == "__main__":
    main()
