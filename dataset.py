import os
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MolFromPDBFile, MolFromMol2File
from torch_geometric.data import InMemoryDataset, Data

from utils import (
    get_binding_affinity,
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
    def __init__(
        self,
        root,
        data_dir,
        affinity_file,
        esm_path=None,
        esm_dim=320,
        transform=None,
        pre_transform=None,
    ):
        self.data_dir = data_dir
        self.affinity_file = affinity_file
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

        self.data, self.slices = torch.load(
            self.processed_paths[0],
            weights_only=False,
        )

    @property
    def processed_file_names(self):
        """
        Important:
        The filename is changed because the graph structure now includes edge_type.
        If we kept the old name, PyG could silently load the old processed dataset.
        """
        return ["ds_pocket_refined_pKd_v2_edge_type.pt"]

    def _load_ligand(self, complex_dir, pdb_id):
        """
        Try loading ligand from MOL2 first, then SDF.
        """
        ligand_path = os.path.join(
            complex_dir,
            f"{pdb_id}_ligand.mol2",
        )

        ligand = None

        if os.path.exists(ligand_path):
            ligand = MolFromMol2File(
                ligand_path,
                sanitize=True,
                removeHs=False,
            )

        if ligand is not None:
            return ligand

        sdf_path = os.path.join(
            complex_dir,
            f"{pdb_id}_ligand.sdf",
        )

        if os.path.exists(sdf_path):
            suppl = Chem.SDMolSupplier(
                sdf_path,
                sanitize=True,
                removeHs=False,
            )

            if suppl is not None and len(suppl) > 0:
                ligand = suppl[0]

        return ligand

    def _load_pocket(self, complex_dir, pdb_id):
        """
        Load protein pocket from PDB file.
        """
        pocket_path = os.path.join(
            complex_dir,
            f"{pdb_id}_pocket.pdb",
        )

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
        map_complex_to_affinity = get_binding_affinity(self.affinity_file)

        folders = [
            folder
            for folder in os.listdir(self.data_dir)
            if not folder.startswith(".")
        ]

        folders = sorted(folders)

        skipped_no_affinity = 0
        skipped_no_ligand = 0
        skipped_no_pocket = 0
        skipped_bad_graph = 0

        for complex_folder_name in tqdm(folders, desc="Processing Pockets"):
            pdb_id = complex_folder_name
            complex_dir = os.path.join(self.data_dir, complex_folder_name)

            if not os.path.isdir(complex_dir):
                continue

            # ---------------------------
            # Check affinity first
            # ---------------------------
            if pdb_id not in map_complex_to_affinity:
                skipped_no_affinity += 1
                continue

            binding_affinity = map_complex_to_affinity[pdb_id]

            # ---------------------------
            # Load ligand and pocket
            # ---------------------------
            ligand = self._load_ligand(complex_dir, pdb_id)

            if ligand is None:
                skipped_no_ligand += 1
                continue

            pocket = self._load_pocket(complex_dir, pdb_id)

            if pocket is None:
                skipped_no_pocket += 1
                continue

            # ---------------------------
            # Build graphs
            # ---------------------------
            ligand_graph = ligand_to_graph(ligand)
            protein_graph = protein_to_graph(pocket)

            if ligand_graph is None or protein_graph is None:
                skipped_bad_graph += 1
                continue

            graph_data = covalent_and_intermolecular_interactions_graph(
                ligand_graph,
                protein_graph,
            )

            if graph_data is None:
                skipped_bad_graph += 1
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

                # Target remains raw pKd.
                # Normalization is done only in Trainer using train statistics.
                y=torch.tensor([binding_affinity], dtype=torch.float32),

                pdb_id=pdb_id,
            )

            data_list.append(data)

        if not data_list:
            raise RuntimeError(
                "No data processed. Check paths, affinity file, and structure files."
            )

        print("\n--- Dataset processing summary ---")
        print(f"Processed complexes:      {len(data_list)}")
        print(f"Skipped no affinity:      {skipped_no_affinity}")
        print(f"Skipped no ligand:        {skipped_no_ligand}")
        print(f"Skipped no pocket:        {skipped_no_pocket}")
        print(f"Skipped bad graph:        {skipped_bad_graph}")

        data, slices = self.collate(data_list)

        torch.save(
            (data, slices),
            self.processed_paths[0],
        )

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