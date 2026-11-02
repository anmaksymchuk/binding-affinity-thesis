import os
import json
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MolFromPDBFile, MolFromMol2File
from torch_geometric.data import InMemoryDataset, Data

from cleansplit import load_complexes_table
from utils import (
    ligand_to_graph,
    protein_to_graph,
    covalent_and_intermolecular_interactions_graph,
)


class ComplexData(Data):
    """
    Custom PyG Data class.

    We have two edge_index tensors:
        edge_index_1: covalent-only graph
        edge_index_2: full graph with covalent + contact edges

    PyG needs to know that both should be incremented by num_nodes
    during batching.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key in ["edge_index_1", "edge_index_2"]:
            return self.num_nodes

        return super().__inc__(key, value, *args, **kwargs)


class Ligand_Protein_Dataset(InMemoryDataset):
    """
    Graphs for all complexes listed in complexes.csv (built by cleansplit.py)
    that have structure files and a label.

    Label filters (censored / approximate) are NOT applied here, so changing
    them in config does not require rebuilding the graphs; main.py selects IDs.

    Next to the graph file, processed/<tag>_index.json records which IDs were
    processed and why the others were skipped.
    """

    def __init__(
        self,
        root,
        data_dir,
        complexes_csv,
        dataset_tag,
        esm_path=None,
        esm_dim=320,
        transform=None,
        pre_transform=None,
    ):
        self.data_dir = data_dir
        self.complexes_csv = complexes_csv
        self.dataset_tag = dataset_tag
        self.esm_dim = esm_dim

        # ---------------------------
        # Load ESM embeddings
        # ---------------------------
        if esm_path is None:
            esm_path = os.path.join(root, "esm_embeddings.pt")

        self.esm_path = esm_path

        if os.path.exists(self.esm_path):
            print(f"--- Loading ESM embeddings from {self.esm_path} ---")
            self.esm_dict = torch.load(self.esm_path, map_location="cpu")
        else:
            print(f"--- Warning: ESM embeddings not found at {self.esm_path} ---")
            self.esm_dict = {}

        super().__init__(root, transform, pre_transform)

        self._load_graphs(self.processed_paths[0])

        with open(self.processed_paths[1]) as f:
            index = json.load(f)

        self.processed_ids = index["processed"]
        self.skipped = index["skipped"]

        self._check_cache_matches_table()

    @property
    def processed_file_names(self):
        return [
            f"graphs_{self.dataset_tag}.pt",
            f"graphs_{self.dataset_tag}_index.json",
        ]

    # ---------------------------
    # Save / load (PyG >= 2.4 has InMemoryDataset.save/load)
    # ---------------------------
    def _save_graphs(self, data_list, path):
        if hasattr(InMemoryDataset, "save"):
            self.save(data_list, path)
        else:
            torch.save(self.collate(data_list), path)

    def _load_graphs(self, path):
        if hasattr(InMemoryDataset, "load"):
            self.load(path)
        else:
            self.data, self.slices = torch.load(path, weights_only=False)

    def _rows_to_process(self):
        table = load_complexes_table(self.complexes_csv)
        return table[table["has_structure"] & table["pk"].notna()]

    def _check_cache_matches_table(self):
        """
        The graph file is cached by name only; make sure it still covers the
        current complexes.csv.
        """
        wanted = set(self._rows_to_process()["pdb_id"])
        known = set(self.processed_ids) | set(self.skipped)
        not_in_cache = wanted - known

        if not_in_cache:
            raise RuntimeError(
                f"{len(not_in_cache)} complexes from {self.complexes_csv} are not in "
                f"the cached graphs {self.processed_paths[0]} "
                f"(e.g. {sorted(not_in_cache)[:5]}). Delete the two files "
                f"{self.processed_file_names} in {self.processed_dir} to rebuild."
            )

    def _load_ligand(self, ligand_path):
        """
        Load ligand from MOL2 or SDF. If the MOL2 file fails to parse,
        try the SDF file next to it.
        """
        ligand = None

        if ligand_path.endswith(".mol2") and os.path.exists(ligand_path):
            ligand = MolFromMol2File(
                ligand_path,
                sanitize=True,
                removeHs=False,
            )

        if ligand is not None:
            return ligand

        sdf_path = ligand_path[: -len(".mol2")] + ".sdf" if ligand_path.endswith(".mol2") else ligand_path

        if os.path.exists(sdf_path):
            suppl = Chem.SDMolSupplier(
                sdf_path,
                sanitize=True,
                removeHs=False,
            )

            if suppl is not None and len(suppl) > 0:
                ligand = suppl[0]

        return ligand

    def _load_pocket(self, pocket_path):
        """
        Load protein pocket from PDB file.
        """
        if not os.path.exists(pocket_path):
            return None

        pocket = MolFromPDBFile(
            pocket_path,
            sanitize=True,
            removeHs=False,
        )

        return pocket

    def process(self):
        data_list = []
        processed_ids = []
        skipped = {}

        rows = self._rows_to_process()

        for row in tqdm(rows.itertuples(), total=len(rows), desc="Processing Pockets"):
            pdb_id = row.pdb_id
            binding_affinity = float(row.pk)

            # ---------------------------
            # Load ligand and pocket
            # ---------------------------
            ligand = self._load_ligand(os.path.join(self.data_dir, row.ligand_file))

            if ligand is None:
                skipped[pdb_id] = "ligand_parse_failed"
                continue

            pocket = self._load_pocket(os.path.join(self.data_dir, row.pocket_file))

            if pocket is None:
                skipped[pdb_id] = "pocket_parse_failed"
                continue

            # ---------------------------
            # Build graphs
            # ---------------------------
            ligand_graph = ligand_to_graph(ligand)
            protein_graph = protein_to_graph(pocket)

            if ligand_graph is None or protein_graph is None:
                skipped[pdb_id] = "no_coordinates"
                continue

            graph_data = covalent_and_intermolecular_interactions_graph(
                ligand_graph,
                protein_graph,
            )

            if graph_data is None:
                skipped[pdb_id] = "bad_graph"
                continue

            # ---------------------------
            # Split graph into two branches
            # ---------------------------
            num_covalent = graph_data["num_covalent_bonds"]

            edge_index_1 = graph_data["edge_index"][:, :num_covalent]
            edge_index_2 = graph_data["edge_index"]

            edge_weight_2 = graph_data["edge_weight"]
            edge_type_2 = graph_data["edge_type"]

            # ---------------------------
            # Create PyG data object
            # ---------------------------
            data = ComplexData(
                x=torch.from_numpy(graph_data["node_feat"]).to(torch.float32),

                # Branch 1: covalent-only graph
                edge_index_1=torch.from_numpy(edge_index_1).to(torch.long),

                # Branch 2: full graph
                edge_index_2=torch.from_numpy(edge_index_2).to(torch.long),
                edge_weight=torch.from_numpy(edge_weight_2).to(torch.float32),
                edge_type=torch.from_numpy(edge_type_2).to(torch.long),

                # Target remains raw pK.
                # Normalization is done only in Trainer using train statistics.
                y=torch.tensor([binding_affinity], dtype=torch.float32),

                pdb_id=pdb_id,
            )

            data_list.append(data)
            processed_ids.append(pdb_id)

        if not data_list:
            raise RuntimeError(
                "No data processed. Check data_dir and complexes.csv."
            )

        print("\n--- Dataset processing summary ---")
        print(f"Complexes in table with structure and label: {len(rows)}")
        print(f"Processed:                                   {len(data_list)}")
        print(f"Skipped:                                     {len(skipped)}")
        for reason in sorted(set(skipped.values())):
            n = sum(1 for r in skipped.values() if r == reason)
            print(f"  {reason:<28}{n}")

        self._save_graphs(data_list, self.processed_paths[0])

        with open(self.processed_paths[1], "w") as f:
            json.dump({"processed": processed_ids, "skipped": skipped}, f, indent=1)

    def get(self, idx):
        data = super().get(idx)

        # ---------------------------
        # Attach ESM vector
        # ---------------------------
        pdb_id = data.pdb_id

        if isinstance(pdb_id, list):
            pdb_id = pdb_id[0]

        if pdb_id in self.esm_dict:
            esm_vec = self.esm_dict[pdb_id]
            has_esm = 1.0
        else:
            esm_vec = torch.zeros(self.esm_dim)
            has_esm = 0.0

        esm_vec = torch.as_tensor(esm_vec, dtype=torch.float32)

        if esm_vec.numel() != self.esm_dim:
            raise ValueError(
                f"ESM vector for {pdb_id} has size {esm_vec.numel()}, "
                f"expected {self.esm_dim}."
            )

        data.esm_vec = esm_vec.view(1, -1)
        data.has_esm = torch.tensor([has_esm], dtype=torch.float32)

        return data