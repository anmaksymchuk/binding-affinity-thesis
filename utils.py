import numpy as np
import torch
from rdkit import Chem
from scipy.spatial import KDTree


# Карта аминокислот
AA_MAP = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}


def atom_to_feature_base(atom):
    """
    Base atom-level features shared by ligand and protein atoms.

    Feature vector length: 11
    """
    hybridization = atom.GetHybridization()

    return [
        atom.GetAtomicNum() * 0.1,
        atom.GetDegree() * 0.2,
        atom.GetTotalNumHs() * 0.2,
        atom.GetImplicitValence() * 0.2,
        float(atom.GetIsAromatic()),
        float(atom.GetFormalCharge()),
        float(hybridization == Chem.rdchem.HybridizationType.SP2),
        float(hybridization == Chem.rdchem.HybridizationType.SP3),
        float(atom.IsInRing()),
        atom.GetMass() * 0.01,
        0.0,  # placeholder for B-factor
    ]


def ligand_atom_to_feature(atom):
    """
    Ligand atom feature vector.

    Final feature layout:
        ligand base features: 11
        protein base placeholder: 11
        amino-acid one-hot placeholder: 21
        ligand indicator: 1

    Total: 44
    """
    base = atom_to_feature_base(atom)

    return np.array(
        base + [0.0] * 11 + [0.0] * 21 + [1.0],
        dtype=np.float32,
    )


def protein_atom_to_feature(atom):
    """
    Protein atom feature vector.

    Final feature layout:
        ligand base placeholder: 11
        protein base features: 11
        amino-acid one-hot: 21
        ligand indicator: 1

    Total: 44
    """
    base = atom_to_feature_base(atom)

    info = atom.GetPDBResidueInfo()
    if info is not None:
        base[10] = info.GetTempFactor() * 0.01

    aa_one_hot = [0.0] * 21

    if info is not None:
        res_name = info.GetResidueName().strip().upper()
        idx = AA_MAP.get(res_name, 20)
        aa_one_hot[idx] = 1.0
    else:
        aa_one_hot[20] = 1.0

    return np.array(
        [0.0] * 11 + base + aa_one_hot + [0.0],
        dtype=np.float32,
    )


def get_positions(mol):
    """
    Extract atom coordinates from the first conformer.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer()

    return np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=np.float32,
    )


def ligand_to_graph(ligand):
    """
    Convert ligand molecule to a graph dictionary.

    Returns
    -------
    dict with:
        node_feat
        edge_index
        node_positions
        num_nodes
    """
    if ligand is None:
        return None

    node_positions = get_positions(ligand)
    if node_positions is None:
        return None

    node_features = np.array(
        [ligand_atom_to_feature(atom) for atom in ligand.GetAtoms()],
        dtype=np.float32,
    )

    edge_list = []
    for bond in ligand.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        edge_list += [(u, v), (v, u)]

    if edge_list:
        edge_index = np.array(edge_list, dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    return {
        "node_feat": node_features,
        "edge_index": edge_index,
        "node_positions": node_positions,
        "num_nodes": node_features.shape[0],
    }


def protein_to_graph(protein):
    """
    Convert protein pocket molecule to a graph dictionary.

    Note:
    RDKit may not always recover perfect covalent connectivity from PDB files.
    This is acceptable for a prototype, but it is worth checking graph statistics.
    """
    if protein is None:
        return None

    node_positions = get_positions(protein)
    if node_positions is None:
        return None

    node_features = np.array(
        [protein_atom_to_feature(atom) for atom in protein.GetAtoms()],
        dtype=np.float32,
    )

    edge_list = []
    for bond in protein.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        edge_list += [(u, v), (v, u)]

    if edge_list:
        edge_index = np.array(edge_list, dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    return {
        "node_feat": node_features,
        "edge_index": edge_index,
        "node_positions": node_positions,
        "num_nodes": node_features.shape[0],
    }


def covalent_and_intermolecular_interactions_graph(
    ligand_graph,
    protein_graph,
    cutoff=5.0,
):
    """
    Build a complex graph with two edge groups:

    1. Covalent edges:
        - ligand internal bonds
        - protein internal bonds
        - edge_type = 0
        - edge_weight = 1.0 placeholder

    2. Intermolecular contact edges:
        - ligand-protein atom pairs within cutoff
        - edge_type = 1
        - edge_weight = actual Euclidean distance

    Returns
    -------
    dict with:
        node_feat
        edge_index
        edge_weight
        edge_type
        num_covalent_bonds
        num_contact_edges
    """
    if ligand_graph is None or protein_graph is None:
        return None

    node_features = np.concatenate(
        [ligand_graph["node_feat"], protein_graph["node_feat"]],
        axis=0,
    )

    num_lig = ligand_graph["num_nodes"]

    # ---------------------------
    # Covalent edges
    # ---------------------------
    ligand_edge_index = ligand_graph["edge_index"]
    protein_edge_index = protein_graph["edge_index"] + num_lig

    cov_edge_index = np.concatenate(
        [ligand_edge_index, protein_edge_index],
        axis=1,
    )

    num_covalent_edges = cov_edge_index.shape[1]

    cov_edge_weight = np.ones(num_covalent_edges, dtype=np.float32)
    cov_edge_type = np.zeros(num_covalent_edges, dtype=np.int64)

    # ---------------------------
    # Intermolecular contact edges
    # ---------------------------
    tree_prot = KDTree(protein_graph["node_positions"])
    contacts = tree_prot.query_ball_point(
        ligand_graph["node_positions"],
        r=cutoff,
    )

    contact_edges = []
    contact_weights = []

    for ligand_atom_idx, protein_neighbors in enumerate(contacts):
        for protein_atom_idx in protein_neighbors:
            dist = np.linalg.norm(
                ligand_graph["node_positions"][ligand_atom_idx]
                - protein_graph["node_positions"][protein_atom_idx]
            )

            u = ligand_atom_idx
            v = protein_atom_idx + num_lig

            contact_edges += [(u, v), (v, u)]
            contact_weights += [dist, dist]

    if contact_edges:
        contact_edge_index = np.array(contact_edges, dtype=np.int64).T
        contact_edge_weight = np.array(contact_weights, dtype=np.float32)
    else:
        contact_edge_index = np.empty((2, 0), dtype=np.int64)
        contact_edge_weight = np.empty((0,), dtype=np.float32)

    num_contact_edges = contact_edge_index.shape[1]
    contact_edge_type = np.ones(num_contact_edges, dtype=np.int64)

    # ---------------------------
    # Full graph
    # ---------------------------
    full_edge_index = np.concatenate(
        [cov_edge_index, contact_edge_index],
        axis=1,
    )

    full_edge_weight = np.concatenate(
        [cov_edge_weight, contact_edge_weight],
        axis=0,
    )

    full_edge_type = np.concatenate(
        [cov_edge_type, contact_edge_type],
        axis=0,
    )

    return {
        "node_feat": node_features,
        "edge_index": full_edge_index,
        "edge_weight": full_edge_weight,
        "edge_type": full_edge_type,
        "num_covalent_bonds": num_covalent_edges,
        "num_contact_edges": num_contact_edges,
    }


def get_binding_affinity(filename):
    """
    Read PDBbind affinity file.

    Expected format:
        PDB_ID ... ... pKd

    The fourth column is used as the affinity target.
    """
    mapping = {}

    with open(filename) as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) >= 4:
                pdb_id = parts[0]
                affinity = float(parts[3])
                mapping[pdb_id] = affinity

    return mapping


def compute_dataset_stats(dataset, train_idx):
    """
    Compute mean and standard deviation of y using only training indices.
    """
    y_values = torch.cat([dataset[i].y for i in train_idx])

    return {
        "mean": y_values.mean().item(),
        "std": y_values.std().item(),
    }

def get_pdb_ids_from_index(filename):
    """
    Read PDB IDs from a PDBbind index file.

    Works for files such as:
        INDEX_refined_data.2016
        INDEX_core_data.2016
        INDEX_core_name.2016

    Returns
    -------
    set[str]
        Lowercase PDB IDs.
    """
    pdb_ids = set()

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) == 0:
                continue

            pdb_id = parts[0].lower()
            pdb_ids.add(pdb_id)

    return pdb_ids