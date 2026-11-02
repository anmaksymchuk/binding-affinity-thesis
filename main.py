import os
import random
import torch
import numpy as np

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
from utils import compute_dataset_stats, get_pdb_ids_from_index

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

    dataset = Ligand_Protein_Dataset(
        root=config.root,
        data_dir=config.data_dir,
        affinity_file=config.affinity_file,
        esm_path=config.esm_path,
        esm_dim=config.esm_dim,
    )

    dataset_size = len(dataset)

    if dataset_size == 0:
        raise RuntimeError("Dataset is empty.")

    print(f"Total complexes: {dataset_size}")

    # ---------------------------
    # Collect processed PDB IDs
    # ---------------------------
    all_indices = np.arange(dataset_size)
    pdb_ids = []

    for i in range(dataset_size):
        data_i = dataset.get(i)
        pdb_id = data_i.pdb_id

        if isinstance(pdb_id, list):
            pdb_id = pdb_id[0]

        pdb_ids.append(str(pdb_id).lower())

    pdb_ids = np.array(pdb_ids)

    # ---------------------------
    # Train / Val / Test split
    # ---------------------------
    if getattr(config, "use_core_test_split", False):
        # CASF-style split:
        #   Test = Core set
        #   Train/Val = Refined set minus Core set

        if not os.path.exists(config.core_index_file):
            raise FileNotFoundError(
                f"Core index file not found: {config.core_index_file}"
            )

        core_ids = get_pdb_ids_from_index(config.core_index_file)

        print("\n--- CASF-style core split ---")
        print(f"Core IDs in index file: {len(core_ids)}")

        processed_ids = set(pdb_ids.tolist())
        core_ids_found = core_ids & processed_ids
        missing_core_ids = core_ids - processed_ids

        print(f"Core IDs found in processed dataset: {len(core_ids_found)}")

        if missing_core_ids:
            print(
                f"Warning: {len(missing_core_ids)} core complexes are absent "
                "from the processed dataset."
            )
            print(f"Missing core IDs: {sorted(missing_core_ids)}")

        test_mask = np.array([pdb_id in core_ids for pdb_id in pdb_ids])

        test_idx = all_indices[test_mask]
        train_val_idx = all_indices[~test_mask]

        if len(test_idx) == 0:
            raise RuntimeError(
                "No core complexes were found in the processed refined dataset. "
                "Check INDEX_core_data.2016 and pdb_id names."
            )

        if len(train_val_idx) == 0:
            raise RuntimeError(
                "Train/val split is empty after removing core complexes."
            )

        # Shuffle only refined-minus-core candidates
        np.random.shuffle(train_val_idx)

        val_size = int(len(train_val_idx) * config.val_split)

        if val_size == 0:
            raise ValueError(
                "Validation split is empty. Increase val_split or check dataset size."
            )

        val_idx = train_val_idx[:val_size]
        train_idx = train_val_idx[val_size:]

        print(
            f"CASF-style split: "
            f"Train={len(train_idx)}, "
            f"Val={len(val_idx)}, "
            f"Test/Core={len(test_idx)}"
        )

    else:
        # Original random split over the full refined set
        print("\n--- Random refined-set split ---")

        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        val_size = int(dataset_size * config.val_split)
        test_size = int(dataset_size * config.test_split)

        if val_size == 0 or test_size == 0:
            raise ValueError(
                "Validation or test split is empty. "
                "Increase dataset size or split ratio."
            )

        val_idx = indices[:val_size]
        test_idx = indices[val_size:val_size + test_size]
        train_idx = indices[val_size + test_size:]

        print(
            f"Random split: "
            f"Train={len(train_idx)}, "
            f"Val={len(val_idx)}, "
            f"Test={len(test_idx)}"
        )

    if len(train_idx) == 0:
        raise ValueError("Training split is empty.")

    # ---------------------------
    # Leakage checks
    # ---------------------------
    train_ids = set(pdb_ids[train_idx])
    val_ids = set(pdb_ids[val_idx])
    test_ids = set(pdb_ids[test_idx])

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    if train_val_overlap:
        raise RuntimeError(
            f"Train/Val overlap found: {sorted(train_val_overlap)[:10]}"
        )

    if train_test_overlap:
        raise RuntimeError(
            f"Train/Test overlap found: {sorted(train_test_overlap)[:10]}"
        )

    if val_test_overlap:
        raise RuntimeError(
            f"Val/Test overlap found: {sorted(val_test_overlap)[:10]}"
        )

    print("Leakage check passed: no PDB ID overlap between train/val/test.")

    # ---------------------------
    # Save split information
    # ---------------------------
    split_info = {
        "seed": config.seed,
        "model_type": model_type,
        "use_core_test_split": getattr(config, "use_core_test_split", False),
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
    }

    split_info_path = os.path.join(config.output_dir, "split_info.pt")
    torch.save(split_info, split_info_path)

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
    test_dataset = dataset[test_idx.tolist()]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

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
    )

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
    # Final evaluation
    # ---------------------------
    print("\n--- Final Evaluation on Test Set ---")

    test_metrics = trainer.eval_step(test_loader)

    test_rmse = test_metrics["rmse"]
    test_mae = test_metrics["mae"]
    test_ci = test_metrics["ci"]
    test_pearson = test_metrics["pearson"]

    print("=" * 50)
    print("TEST RESULTS (pKd Scale)")
    print(f"RMSE:    {test_rmse:.4f}")
    print(f"MAE:     {test_mae:.4f}")
    print(f"CI:      {test_ci:.4f}")
    print(f"Pearson: {test_pearson:.4f}")
    print("=" * 50)

    # ---------------------------
    # Save artifacts
    # ---------------------------
    torch.save(best_model.state_dict(), config.final_model_path)
    torch.save(y_stats, config.y_stats_path)
    torch.save(history, config.history_path)
    torch.save(test_metrics, config.test_metrics_path)

    print("\nSaved files:")
    print(f"  - {config.best_model_path}")
    print(f"  - {config.final_model_path}")
    print(f"  - {config.y_stats_path}")
    print(f"  - {config.history_path}")
    print(f"  - {config.test_metrics_path}")
    print(f"  - {split_info_path}")
    print("\nAll saved. Done!")


if __name__ == "__main__":
    main()