"""
Evaluate a saved checkpoint on the CASF test sets without training.

Useful when a Colab session ended before main.py reached the test step:
the best checkpoint (best_model_rmse.pt) and split_info.json are written
during training. Run from the repo root with the same config.py as the run:

    python tools/evaluate.py            # uses config.output_dir
    python tools/evaluate.py <output_dir>

Currently supports model_type = "gnn_only".
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from config import Config
from dataset import Ligand_Protein_Dataset
from model import GNN_Only_Predictor
from trainer import Trainer


def main():
    c = Config()
    out_dir = sys.argv[1] if len(sys.argv) > 1 else c.output_dir

    with open(os.path.join(out_dir, "split_info.json")) as f:
        split = json.load(f)

    if split["model_type"] != "gnn_only":
        raise NotImplementedError(f"model_type {split['model_type']} is not supported yet")

    kwargs = dict(
        root=c.root, data_dir=c.data_dir, complexes_csv=c.complexes_csv,
        dataset_tag=split["dataset_tag"], esm_path=c.esm_path, esm_dim=c.esm_dim,
    )
    for name in ("graph_shard_size", "pocket_hydrogens"):
        if hasattr(c, name):
            kwargs["shard_size" if name == "graph_shard_size" else name] = getattr(c, name)
    ds = Ligand_Protein_Dataset(**kwargs)

    id_to_idx = {}
    for i in range(len(ds)):
        pdb_id = ds.get(i).pdb_id
        id_to_idx[pdb_id[0] if isinstance(pdb_id, list) else pdb_id] = i

    # Target statistics: saved file if present, otherwise recomputed from train IDs
    y_stats_path = os.path.join(out_dir, "y_stats.pt")
    if os.path.exists(y_stats_path):
        y_stats = torch.load(y_stats_path)
    else:
        y = torch.cat([ds.get(id_to_idx[i]).y for i in split["train_ids"]])
        y_stats = {"mean": y.mean().item(), "std": y.std().item()}
    print(f"y_stats: mean {y_stats['mean']:.4f}, std {y_stats['std']:.4f}")

    model = GNN_Only_Predictor(
        in_channels=c.in_channels, num_gnn_layers=c.num_gnn_layers,
        linear_out_channels=c.linear_out_channels, hidden_dim=c.hidden_dim,
        dropout_gnn=c.dropout_gnn, dropout_mlp=c.dropout_mlp,
    ).to(c.device)
    ckpt = os.path.join(out_dir, "best_model_rmse.pt")
    model.load_state_dict(torch.load(ckpt, map_location=c.device, weights_only=True))
    print(f"Loaded {ckpt}")

    trainer = Trainer(
        model=model, device=c.device, criterion=torch.nn.HuberLoss(),
        optimizer=torch.optim.AdamW(model.parameters()), y_stats=y_stats,
    )

    metrics_all, frames = {}, []
    for name, ids in split["test_ids"].items():
        loader = DataLoader(ds[[id_to_idx[i] for i in ids]], batch_size=c.test_batch_size)
        metrics, preds = trainer.eval_step(loader, return_predictions=True)
        metrics_all[name] = {**metrics, "n": len(ids)}
        df = pd.DataFrame(preds)
        df.insert(0, "test_set", name)
        frames.append(df)

    print(f"{'set':<16}{'n':>5}{'RMSE':>9}{'MAE':>9}{'CI':>9}{'Pearson':>10}")
    for name, m in metrics_all.items():
        print(f"{name:<16}{m['n']:>5}{m['rmse']:>9.4f}{m['mae']:>9.4f}{m['ci']:>9.4f}{m['pearson']:>10.4f}")

    torch.save(metrics_all, os.path.join(out_dir, "test_metrics.pt"))
    pd.concat(frames).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    # Validation predictions (for error vs. similarity analysis)
    loader = DataLoader(ds[[id_to_idx[i] for i in split["val_ids"]]], batch_size=c.test_batch_size)
    metrics, preds = trainer.eval_step(loader, return_predictions=True)
    print(f"{'val':<16}{len(split['val_ids']):>5}{metrics['rmse']:>9.4f}{metrics['mae']:>9.4f}"
          f"{metrics['ci']:>9.4f}{metrics['pearson']:>10.4f}")
    df = pd.DataFrame(preds)
    df.insert(0, "test_set", "val")
    df.to_csv(os.path.join(out_dir, "val_predictions.csv"), index=False)

    print(f"Saved test_metrics.pt, test_predictions.csv and val_predictions.csv to {out_dir}")


if __name__ == "__main__":
    main()
