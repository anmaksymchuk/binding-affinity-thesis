import os
import json
import math
import bisect
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MolFromPDBFile, MolFromMol2File
from torch_geometric.data import Dataset, InMemoryDataset, Data

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


class _GraphShard(InMemoryDataset):
    """
    One collated shard of graphs, loaded from disk (never processed here).
    """

    def __init__(self, path):
        super().__init__(root=None)

        if hasattr(InMemoryDataset, "load"):
            self.load(path)
        else:
            self.data, self.slices = torch.load(path, weights_only=False)


class Ligand_Protein_Dataset(Dataset):
    """
    Graphs for all complexes listed in complexes.csv (built by cleansplit.py)
    that have structure files and a label.

    Label filters (censored / approximate) are NOT applied here, so changing
    them in config does not require rebuilding the graphs; main.py selects IDs.

    Graphs are built and saved in shards of `shard_size` complexes:
        processed/graphs_<tag>_shardNNN.pt    collated graphs
        processed/graphs_<tag>_shardNNN.json  IDs of the shard and why some were skipped
    Collating all ~17k graphs at once needs about twice the dataset size in RAM
    and crashed Colab; shards keep the peak small. If processing is
    interrupted, finished shards are reused on the next run.
    """

    def __init__(
        self,
        root,
        data_dir,
        complexes_csv,
        dataset_tag,
        esm_path=None,
        esm_dim=320,
        shard_size=1000,
        transform=None,
        pre_transform=None,
    ):
        self.data_dir = data_dir
        self.complexes_csv = complexes_csv
        self.dataset_tag = dataset_tag
        self.esm_dim = esm_dim
        self.shard_size = shard_size

        self._rows = self._rows_to_process()
        self.n_shards = max(1, math.ceil(len(self._rows) / shard_size))

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

        self._load_shards()

    # ---------------------------
    # Files
    # ---------------------------
    def _shard_base(self, k):
        return f"graphs_{self.dataset_tag}_shard{k:03d}"

    def _shard_paths(self, k):
        base = os.path.join(self.processed_dir, self._shard_base(k))
        return base + ".pt", base + ".json"

    def _shard_rows(self, k):
        return self._rows.iloc[k * self.shard_size:(k + 1) * self.shard_size]

    @property
    def processed_file_names(self):
        names = []
        for k in range(self.n_shards):
            names += [self._shard_base(k) + ".pt", self._shard_base(k) + ".json"]
        return names

    def _rows_to_process(self):
        table = load_complexes_table(self.complexes_csv)
        rows = table[table["has_structure"] & table["pk"].notna()]
        return rows.sort_values("pdb_id").reset_index(drop=True)

    def _load_shards(self):
        """
        Load all shards and check that they match the current complexes.csv
        (the cache is keyed by file name only).
        """
        self._shards = []
        self._offsets = [0]
        self.processed_ids = []
        self.skipped = {}

        for k in range(self.n_shards):
            pt_path, json_path = self._shard_paths(k)

            with open(json_path) as f:
                info = json.load(f)

            expected = list(self._shard_rows(k)["pdb_id"])
            if info["ids"] != expected:
                raise RuntimeError(
                    f"Cached shard {json_path} does not match {self.complexes_csv} "
                    f"(the table changed after the graphs were built). Delete "
                    f"{self.processed_dir}/graphs_{self.dataset_tag}_shard* to rebuild."
                )

            self.processed_ids += info["processed"]
            self.skipped.update(info["skipped"])

            if info["processed"]:
                shard = _GraphShard(pt_path)
                if len(shard) != len(info["processed"]):
                    raise RuntimeError(f"{pt_path} is incomplete. Delete it to rebuild.")
                self._shards.append(shard)
                self._offsets.append(self._offsets[-1] + len(shard))

    def len(self):
        return self._offsets[-1]

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

    def _build_graph(self, row):
        """
        Returns (ComplexData, "") or (None, reason).
        """
        pdb_id = row.pdb_id
        binding_affinity = float(row.pk)

        # ---------------------------
        # Load ligand and pocket
        # ---------------------------
        ligand = self._load_ligand(os.path.join(self.data_dir, row.ligand_file))

        if ligand is None:
            return None, "ligand_parse_failed"

        pocket = self._load_pocket(os.path.join(self.data_dir, row.pocket_file))

        if pocket is None:
            return None, "pocket_parse_failed"

        # ---------------------------
        # Build graphs
        # ---------------------------
        ligand_graph = ligand_to_graph(ligand)
        protein_graph = protein_to_graph(pocket)

        if ligand_graph is None or protein_graph is None:
            return None, "no_coordinates"

        graph_data = covalent_and_intermolecular_interactions_graph(
            ligand_graph,
            protein_graph,
        )

        if graph_data is None:
            return None, "bad_graph"

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

        return data, ""

    def process(self):
        n_total = 0
        n_done_before = 0

        for k in range(self.n_shards):
            pt_path, json_path = self._shard_paths(k)
            rows = self._shard_rows(k)
            n_total += len(rows)

            if os.path.exists(pt_path) and os.path.exists(json_path):
                n_done_before += 1
                continue

            data_list = []
            processed_ids = []
            skipped = {}

            for row in tqdm(
                rows.itertuples(),
                total=len(rows),
                desc=f"Processing shard {k + 1}/{self.n_shards}",
            ):
                data, reason = self._build_graph(row)

                if data is None:
                    skipped[row.pdb_id] = reason
                    continue

                data_list.append(data)
                processed_ids.append(row.pdb_id)

            if data_list:
                if hasattr(InMemoryDataset, "save"):
                    InMemoryDataset.save(data_list, pt_path)
                else:
                    torch.save(InMemoryDataset.collate(data_list), pt_path)
            else:
                torch.save(None, pt_path)

            del data_list

            # The .json is written last: if it exists, the .pt is complete.
            with open(json_path, "w") as f:
                json.dump(
                    {"ids": list(rows["pdb_id"]), "processed": processed_ids, "skipped": skipped},
                    f,
                    indent=1,
                )

        # ---------------------------
        # Summary over all shards
        # ---------------------------
        n_processed = 0
        skipped = {}
        for k in range(self.n_shards):
            with open(self._shard_paths(k)[1]) as f:
                info = json.load(f)
            n_processed += len(info["processed"])
            skipped.update(info["skipped"])

        if n_processed == 0:
            raise RuntimeError(
                "No data processed. Check data_dir and complexes.csv."
            )

        print("\n--- Dataset processing summary ---")
        if n_done_before:
            print(f"Shards reused from a previous run:           {n_done_before}/{self.n_shards}")
        print(f"Complexes in table with structure and label: {n_total}")
        print(f"Processed:                                   {n_processed}")
        print(f"Skipped:                                     {len(skipped)}")
        for reason in sorted(set(skipped.values())):
            n = sum(1 for r in skipped.values() if r == reason)
            print(f"  {reason:<28}{n}")

    def get(self, idx):
        k = bisect.bisect_right(self._offsets, idx) - 1
        data = self._shards[k].get(idx - self._offsets[k])

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