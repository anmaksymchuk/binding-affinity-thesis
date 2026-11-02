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

    # PDBbind v2020.R1, package "Protein-ligand complex structures", unpacked.
    # Any folder layout works: files <id>_ligand.mol2/.sdf and <id>_pocket.pdb
    # are found recursively by cleansplit.py.
    data_dir = os.path.join(project_dir, "PDBbind_v2020R1")

    # Index file with affinities from the same R1 download.
    # If this exact path does not exist, cleansplit.py looks for
    # INDEX_general_PL* under data_dir and project_dir.
    r1_index_file = os.path.join(data_dir, "index", "INDEX_general_PL.2020R1.lst")

    # "r1_index"  - labels from the R1 index file (main protocol)
    # "gems_dict" - original v2020 labels shipped with CleanSplit
    label_source = "r1_index"

    # Name of the cached graph file in processed/. Change it (or delete the
    # file) whenever graph building or the label source changes.
    dataset_tag = f"v2020R1_cleansplit_{label_source}"

    # CleanSplit ID lists (copied from the GEMS repo, see splits/cleansplit/README.md)
    split_dir = os.path.join(project_dir, "splits", "cleansplit")
    split_file = os.path.join(split_dir, "PDBbind_data_split_cleansplit.json")
    fold_file_pattern = "PDBbind_cleansplit_train_val_split_f{fold}.json"
    gems_data_dict = os.path.join(split_dir, "PDBbind_data_dict.json")

    # Master table written by `python cleansplit.py`
    complexes_csv = os.path.join(project_dir, "data", "complexes.csv")

    esm_path = os.path.join(project_dir, "esm_embeddings.pt")
    processed_dir = os.path.join(root, "processed")


    # ============================================================
    # Dataset split
    # ============================================================
    # Train/val come from CleanSplit "train"; test sets are the CASF lists.
    #   "random":    val = random `val_split` fraction of CleanSplit train
    #                (shuffled with `seed`)
    #   "gems_fold": val = "validation" list of GEMS fold `fold` (0..4),
    #                train = the rest of CleanSplit train
    val_mode = "random"
    fold = 0
    val_split = 0.15

    test_sets = ["casf2016", "casf2016_indep", "casf2013", "casf2013_indep"]
    main_test_set = "casf2016"

    # Label filters (applied in cleansplit.py and main.py)
    drop_censored_labels = True   # drop <, <=, >, >=
    keep_approx_labels = True     # keep ~

    # One output folder per (model, validation split, seed), so runs do not overwrite each other
    val_tag = f"fold{fold}" if val_mode == "gems_fold" else f"val{int(val_split * 100)}"
    output_dir = os.path.join(
        project_dir, "outputs", f"{experiment_name}_{val_tag}_seed{seed}"
    )
    os.makedirs(output_dir, exist_ok=True)

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
    test_predictions_path = os.path.join(output_dir, "test_predictions.csv")
    split_info_path = os.path.join(output_dir, "split_info.json")