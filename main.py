import os
import json
import random
import torch
import numpy as np
import pandas as pd

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from model import (
    Binding_Affinity_Predictor,
    ESM_Only_Predictor,
    GNN_Only_Predictor,
    GNN_Only_VQC_Predictor,
)
from dataset import Ligand_Protein_Dataset
from trainer import Trainer
from config import Config
from utils import compute_dataset_stats
from cleansplit import load_complexes_table, make_split, sha256_of, fold_file

from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    config = Config()
    model_type = getattr(config, "model_type", "full").lower()

    set_seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("Binding Affinity Prediction Training")
    print("=" * 60)
    print(f"Project dir: {config.project_dir}")
    print(f"Device:      {config.device}")
    print(f"Output dir:  {config.output_dir}")
    print(f"Seed:        {config.seed}")
    print(f"Model type:  {model_type}")

    if model_type == "full":
        print(f"FiLM scale:  {config.film_scale}")
    elif model_type == "gnn_vqc":
        print(f"QNN layers:  {config.qnn_layers}")
        print(f"Qubits:      {config.n_qubits}")

    # ---------------------------
    # Dataset initialization
    # ---------------------------
    print(f"\n--- Loading Dataset from {config.data_dir} ---")

    table = load_complexes_table(config.complexes_csv)

    dataset = Ligand_Protein_Dataset(
        root=config.root,
        data_dir=config.data_dir,
        complexes_csv=config.complexes_csv,
        dataset_tag=config.dataset_tag,
        shard_size=config.graph_shard_size,
        esm_path=config.esm_path,
        esm_dim=config.esm_dim,
    )

    dataset_size = len(dataset)

    if dataset_size == 0:
        raise RuntimeError("Dataset is empty.")

    print(f"Total processed complexes: {dataset_size}")

    # ---------------------------
    # Collect processed PDB IDs
    # ---------------------------
    pdb_ids = []
    y_cached = []

    for i in range(dataset_size):
        data_i = dataset.get(i)
        pdb_id = data_i.pdb_id

        if isinstance(pdb_id, list):
            pdb_id = pdb_id[0]

        pdb_ids.append(str(pdb_id).lower())
        y_cached.append(float(data_i.y.view(-1)[0]))

    pdb_ids = np.array(pdb_ids)
    id_to_idx = {pdb_id: i for i, pdb_id in enumerate(pdb_ids)}

    if len(id_to_idx) != len(pdb_ids):
        raise RuntimeError("Duplicate PDB IDs in the processed dataset.")

    # Labels in the cached graphs must match complexes.csv
    pk_table = dict(zip(table["pdb_id"], table["pk"]))
    stale = [
        str(pdb_id) for pdb_id, y in zip(pdb_ids, y_cached)
        if not abs(pk_table.get(pdb_id, np.nan) - y) < 1e-4
    ]
    if stale:
        raise RuntimeError(
            f"{len(stale)} cached labels differ from {config.complexes_csv} "
            f"(e.g. {stale[:5]}). Delete processed/graphs_{config.dataset_tag}* "
            "to rebuild the graphs."
        )

    # ---------------------------
    # CleanSplit: Train / Val / Tests by PDB ID
    # ---------------------------
    split = make_split(table, set(pdb_ids), config, skipped=dataset.skipped)

    train_ids = split["train_ids"]
    val_ids = split["val_ids"]
    test_ids = split["test_ids"]

    if config.main_test_set not in test_ids:
        raise ValueError(
            f"main_test_set={config.main_test_set} is not in test_sets={config.test_sets}"
        )

    train_idx = np.array([id_to_idx[i] for i in train_ids])
    val_idx = np.array([id_to_idx[i] for i in val_ids])
    test_idx = {name: np.array([id_to_idx[i] for i in ids]) for name, ids in test_ids.items()}

    print("\n--- CleanSplit ---")
    val_desc = (
        f"GEMS fold {config.fold}" if config.val_mode == "gems_fold"
        else f"random {config.val_split:.0%} (seed {config.seed})"
    )
    print(f"Validation: {val_desc}; labels: {config.label_source}")
    print(f"Train={len(train_idx)}, Val={len(val_idx)}")
    for name, idx in test_idx.items():
        print(f"Test {name}: {len(idx)}")

    for name, c in split["counts"].items():
        if c["excluded_by_reason"]:
            print(f"Excluded from {name}: {c['excluded_by_reason']}")

    print("Leakage check passed: no PDB ID overlap between train/val and CASF sets.")

    # ---------------------------
    # Save split information
    # ---------------------------
    split_info = {
        "seed": config.seed,
        "model_type": model_type,
        "protocol": "PDBbind v2020.R1 + CleanSplit",
        "label_source": config.label_source,
        "drop_censored_labels": config.drop_censored_labels,
        "keep_approx_labels": config.keep_approx_labels,
        "val_mode": config.val_mode,
        "fold": config.fold if config.val_mode == "gems_fold" else None,
        "val_split": config.val_split if config.val_mode == "random" else None,
        "main_test_set": config.main_test_set,
        "split_file": os.path.relpath(config.split_file, config.project_dir),
        "split_file_sha256": sha256_of(config.split_file),
        "fold_file_sha256": (
            sha256_of(fold_file(config, config.fold))
            if config.val_mode == "gems_fold" else None
        ),
        "complexes_csv_sha256": sha256_of(config.complexes_csv),
        "dataset_tag": config.dataset_tag,
        "counts": split["counts"],
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "excluded": split["excluded"],
    }

    split_info_path = config.split_info_path

    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=1)

    # ---------------------------
    # Target statistics
    # ---------------------------
    # Important: compute mean/std only on the training split.
    y_stats = compute_dataset_stats(dataset, train_idx)

    print(
        f"Train target statistics: "
        f"Mean={y_stats['mean']:.4f}, "
        f"Std={y_stats['std']:.4f}"
    )

    if y_stats["std"] < 1e-8:
        raise ValueError("Target std is too small. Cannot normalize y safely.")

    # ---------------------------
    # DataLoaders
    # ---------------------------
    train_dataset = dataset[train_idx.tolist()]
    val_dataset = dataset[val_idx.tolist()]
    test_datasets = {name: dataset[idx.tolist()] for name, idx in test_idx.items()}

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        # BatchNorm fails on a last batch of size 1; with the new train size
        # this can happen, so the incomplete last batch is dropped.
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    test_loaders = {
        name: DataLoader(
            ds,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        for name, ds in test_datasets.items()
    }

    # ---------------------------
    # Model
    # ---------------------------
    if model_type == "full":
        model = Binding_Affinity_Predictor(
            in_channels=config.in_channels,
            num_gnn_layers=config.num_gnn_layers,
            linear_out_channels=config.linear_out_channels,
            esm_dim=config.esm_dim,
            hidden_dim=config.hidden_dim,
            dropout_gnn=config.dropout_gnn,
            dropout_mlp=config.dropout_mlp,
            film_scale=config.film_scale,
        )

    elif model_type == "gnn_only":
        model = GNN_Only_Predictor(
            in_channels=config.in_channels,
            num_gnn_layers=config.num_gnn_layers,
            linear_out_channels=config.linear_out_channels,
            hidden_dim=config.hidden_dim,
            dropout_gnn=config.dropout_gnn,
            dropout_mlp=config.dropout_mlp,
        )

    elif model_type == "gnn_vqc":
        model = GNN_Only_VQC_Predictor(
            in_channels=config.in_channels,
            num_gnn_layers=config.num_gnn_layers,
            hidden_dim=config.hidden_dim,
            dropout_gnn=config.dropout_gnn,
            dropout_mlp=config.dropout_mlp,
            n_qubits=config.n_qubits,
            qnn_layers=config.qnn_layers,
            q_device=config.q_device,
        )

    elif model_type == "esm_only":
        model = ESM_Only_Predictor(
            linear_out_channels=config.linear_out_channels,
            esm_dim=config.esm_dim,
            dropout_mlp=config.dropout_mlp,
        )

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            "Use 'full', 'gnn_only', 'gnn_vqc', or 'esm_only'."
        )

    model = model.to(config.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {num_params:,}")

    # ---------------------------
    # Optimizer
    # ---------------------------

    if model_type == "gnn_vqc":
        quantum_params = []
        classical_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if "qlayer" in name:
                quantum_params.append(param)
            else:
                classical_params.append(param)

        optimizer = AdamW(
            [
                {
                    "params": classical_params,
                    "lr": config.learning_rate,
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": quantum_params,
                    "lr": config.quantum_learning_rate,
                    "weight_decay": 0.01,
                },
            ]
        )

    else:
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    # ---------------------------
    # Criterion
    # ---------------------------
    criterion_type = getattr(config, "criterion_type", "huber").lower()

    if criterion_type == "huber":
        criterion = torch.nn.HuberLoss()
    elif criterion_type == "mse":
        criterion = torch.nn.MSELoss()
    elif criterion_type == "mae":
        criterion = torch.nn.L1Loss()
    else:
        raise ValueError(
            f"Unknown criterion_type: {criterion_type}. "
            "Use 'huber', 'mse', or 'mae'."
        )

    print(f"Criterion: {criterion.__class__.__name__}")

    # ---------------------------
    # Scheduler
    # ---------------------------
    use_scheduler = getattr(config, "use_scheduler", True)

    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
        print("Scheduler: ReduceLROnPlateau")
    else:
        scheduler = None
        print("Scheduler: disabled")

    # ---------------------------
    # Trainer
    # ---------------------------
    trainer = Trainer(
        model=model,
        device=config.device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        y_stats=y_stats,
        grad_clip_norm=config.grad_clip_norm,
        checkpoint_path=config.best_model_path,
        use_amp=getattr(config, "use_amp", False) and model_type != "gnn_vqc",
    )

    print(f"Mixed precision (AMP): {trainer.use_amp}")

    # ---------------------------
    # Training
    # ---------------------------
    print("\n--- Starting Training with Z-Score Target Normalization ---")

    if getattr(config, "detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)

    best_model, history = trainer.train(
        epochs=config.num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stop=config.early_stop,
        patience=config.patience,
    )

    # ---------------------------
    # Final evaluation on all CASF test sets
    # ---------------------------
    print("\n--- Final Evaluation on Test Sets ---")

    test_metrics = {}
    prediction_frames = []

    for name, loader in test_loaders.items():
        metrics, preds = trainer.eval_step(loader, return_predictions=True)
        test_metrics[name] = {**metrics, "n": len(preds["pdb_id"])}

        df = pd.DataFrame(preds)
        df.insert(0, "test_set", name)
        prediction_frames.append(df)

    print("=" * 66)
    print(f"TEST RESULTS (pK scale), main test set: {config.main_test_set}")
    print(f"{'set':<16}{'n':>5}{'RMSE':>9}{'MAE':>9}{'CI':>9}{'Pearson':>10}")
    for name, m in test_metrics.items():
        print(
            f"{name:<16}{m['n']:>5}{m['rmse']:>9.4f}{m['mae']:>9.4f}"
            f"{m['ci']:>9.4f}{m['pearson']:>10.4f}"
        )
    print("=" * 66)

    # ---------------------------
    # Save artifacts
    # ---------------------------
    torch.save(best_model.state_dict(), config.final_model_path)
    torch.save(y_stats, config.y_stats_path)
    torch.save(history, config.history_path)
    torch.save(test_metrics, config.test_metrics_path)
    pd.concat(prediction_frames).to_csv(config.test_predictions_path, index=False)

    print("\nSaved files:")
    print(f"  - {config.best_model_path}")
    print(f"  - {config.final_model_path}")
    print(f"  - {config.y_stats_path}")
    print(f"  - {config.history_path}")
    print(f"  - {config.test_metrics_path}")
    print(f"  - {config.test_predictions_path}")
    print(f"  - {split_info_path}")
    print("\nAll saved. Done!")


if __name__ == "__main__":
    main()