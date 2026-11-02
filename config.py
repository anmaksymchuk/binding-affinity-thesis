import os
import torch


class Config:
    # ============================================================
    # Experiment
    # ============================================================
    model_type = "gnn_vqc"      # "full", "gnn_only", "gnn_vqc", or "esm_only"
    #experiment_name = model_type
    experiment_name = f"{model_type}"

    # ============================================================
    # Environment
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # ============================================================
    # Paths
    # ============================================================
    project_dir = os.getcwd()
    root = project_dir

    data_dir = os.path.join(project_dir, "refined_set")

    affinity_file = os.path.join(
        project_dir,
        "PDBbind_2016",
        "index",
        "INDEX_refined_data.2016",
    )

    core_index_file = os.path.join(
        project_dir,
        "PDBbind_2016",
        "index",
        "INDEX_core_data.2016",
    )

    esm_path = os.path.join(project_dir, "esm_embeddings.pt")
    processed_dir = os.path.join(root, "processed")

    output_dir = os.path.join(project_dir, "outputs", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # Dataset split
    # ============================================================
    use_core_test_split = True

    # For CASF-style split:
    #   Test = Core set
    #   Train/Val = Refined set minus Core set
    val_split = 0.15

    # Only used if use_core_test_split = False
    test_split = 0.15

    # ============================================================
    # DataLoader settings
    # ============================================================
    train_batch_size = 32
    val_batch_size = 32
    test_batch_size = 32

    num_workers = 0
    pin_memory = True if device == "cuda" else False

    # ============================================================
    # Architecture
    # ============================================================
    in_channels = 44
    esm_dim = 320

    hidden_dim = 128
    num_gnn_layers = 3

    linear_out_channels = [256, 128, 64]

    dropout_gnn = 0.1
    dropout_mlp = 0.3

    # Used only by the full hybrid model.
    # Ignored by gnn_only and esm_only.
    film_scale = 0.0

    # ============================================================
    # Quantum head
    # ============================================================
    n_qubits = 10
    qnn_layers = 8
    q_device = "default.qubit"
    quantum_learning_rate = 5e-5

    # ============================================================
    # Optimization
    # ============================================================
    learning_rate = 1e-4
    weight_decay = 1e-3

    criterion_type = "huber"
    grad_clip_norm = 1.0

    # ============================================================
    # Scheduler
    # ============================================================
    use_scheduler = True
    scheduler_factor = 0.5
    scheduler_patience = 10

    # ============================================================
    # Training
    # ============================================================
    num_epochs = 200

    early_stop = True
    patience = 30

    detect_anomaly = False

    # ============================================================
    # Saved files
    # ============================================================
    best_model_path = os.path.join(output_dir, "best_model_rmse.pt")
    final_model_path = os.path.join(output_dir, "model_final_pKd.pt")
    y_stats_path = os.path.join(output_dir, "y_stats.pt")
    history_path = os.path.join(output_dir, "training_history.pt")
    test_metrics_path = os.path.join(output_dir, "test_metrics.pt")