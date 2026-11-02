import torch
import numpy as np
import pennylane as qml

from torch.nn import (
    ModuleList,
    Linear,
    LeakyReLU,
    BatchNorm1d,
    Dropout,
    Sequential,
    Parameter,
    LayerNorm,
)
from torch_geometric.nn import global_mean_pool

from Graph_Attention_Layer import Gate_Augmented_GATv2_Layer


class Binding_Affinity_Predictor(torch.nn.Module):
    """
    Hybrid dual-branch GNN + ESM model for protein-ligand binding affinity prediction.

    Branch 1:
        Covalent-only graph.

    Branch 2:
        Full graph with covalent + intermolecular contact edges.

    Additional components:
        - ESM-based FiLM conditioning
        - Virtual node for global graph-level communication
        - Final MLP regression head using graph features + virtual node + ESM
    """

    def __init__(
        self,
        in_channels,
        num_gnn_layers,
        linear_out_channels,
        esm_dim=320,
        hidden_dim=128,
        dropout_gnn=0.1,
        dropout_mlp=0.3,
        film_scale=0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.esm_dim = esm_dim
        self.film_scale = film_scale

        # ---------------------------
        # Input projection
        # ---------------------------
        self.input_transform = Linear(in_channels, self.hidden_dim)

        # ---------------------------
        # ESM-based FiLM conditioning
        # ---------------------------
        self.esm_to_gamma = Linear(esm_dim, self.hidden_dim)
        self.esm_to_beta = Linear(esm_dim, self.hidden_dim)

        # ---------------------------
        # Shared GNN layers
        # ---------------------------
        self.gnn_layers = ModuleList()

        # Separate BatchNorms for the two branches.
        # This avoids mixing running statistics from covalent-only
        # and full-graph streams.
        self.batch_norms_1 = ModuleList()
        self.batch_norms_2 = ModuleList()

        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                Gate_Augmented_GATv2_Layer(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    heads=4,
                    dropout=dropout_gnn,
                    save_attention=False,
                )
            )

            self.batch_norms_1.append(BatchNorm1d(self.hidden_dim))
            self.batch_norms_2.append(BatchNorm1d(self.hidden_dim))

        self.act = LeakyReLU(0.01)

        # ---------------------------
        # Virtual node
        # ---------------------------
        self.vn_embedding = Parameter(torch.zeros(1, self.hidden_dim))

        self.vn_mlp = ModuleList()
        self.vn_gate = ModuleList()
        self.vn_norm = ModuleList()

        for _ in range(num_gnn_layers):
            self.vn_mlp.append(
                Sequential(
                    Linear(self.hidden_dim, self.hidden_dim),
                    LeakyReLU(0.01),
                    Linear(self.hidden_dim, self.hidden_dim),
                )
            )

            # Gate for mixing virtual-node information into node states
            self.vn_gate.append(Linear(self.hidden_dim * 2, 1))
            self.vn_norm.append(LayerNorm(self.hidden_dim))

        # ---------------------------
        # Final MLP
        # ---------------------------
        # graph_repr: x_1 + x_2 = 128 + 128
        # vn_state: 128
        # esm_vec: 320
        #
        # Total = 128 * 2 + 128 + 320 = 704
        curr_dim = (self.hidden_dim * 2) + self.hidden_dim + esm_dim

        self.mlp = ModuleList()

        for out_dim in linear_out_channels:
            self.mlp.append(
                Sequential(
                    Linear(curr_dim, out_dim),
                    BatchNorm1d(out_dim),
                    LeakyReLU(0.01),
                    Dropout(dropout_mlp),
                )
            )
            curr_dim = out_dim

        self.out = Linear(curr_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset all trainable modules.

        This correctly reaches modules inside Sequential and ModuleList.
        """
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

        torch.nn.init.xavier_uniform_(self.vn_embedding)

    def forward(self, data):
        x = data.x
        edge_index_1 = data.edge_index_1
        edge_index_2 = data.edge_index_2
        edge_weight = data.edge_weight
        edge_type = data.edge_type
        esm_vec = data.esm_vec
        batch = data.batch

        # Safety for a single unbatched graph
        if esm_vec.dim() == 1:
            esm_vec = esm_vec.unsqueeze(0)

        # ---------------------------
        # 1. Input transformation
        # ---------------------------
        x = self.input_transform(x)

        # ---------------------------
        # 2. ESM FiLM conditioning
        # ---------------------------
        gamma = self.film_scale * torch.tanh(self.esm_to_gamma(esm_vec))
        beta = self.film_scale * self.esm_to_beta(esm_vec)

        gamma_node = gamma.index_select(0, batch)
        beta_node = beta.index_select(0, batch)

        # Residual FiLM
        x = x * (1.0 + gamma_node) + beta_node

        # ---------------------------
        # 3. Virtual-node initialization
        # ---------------------------
        num_graphs = int(batch.max().item()) + 1
        vn_state = self.vn_embedding.expand(num_graphs, -1)

        # ---------------------------
        # 4. Dual-path GNN
        # ---------------------------
        # x_1: covalent-only branch
        # x_2: full-graph branch
        x_1 = x
        x_2 = x

        for i in range(self.num_gnn_layers):
            # ---------------------------
            # VN -> nodes
            # ---------------------------
            vn_msg = vn_state.index_select(0, batch)

            # Use full-graph branch as context for VN gate
            z = torch.sigmoid(
                self.vn_gate[i](
                    torch.cat([x_2, vn_msg], dim=-1)
                )
            )

            x_1 = (1.0 - z) * x_1 + z * vn_msg
            x_2 = (1.0 - z) * x_2 + z * vn_msg

            # ---------------------------
            # Message passing
            # ---------------------------
            # Branch 1 has only the covalent graph,
            # so no edge attributes are needed.
            x_1 = self.gnn_layers[i](
                x=x_1,
                edge_index=edge_index_1,
            )

            # Branch 2 uses the full graph with explicit
            # edge_weight and edge_type.
            x_2 = self.gnn_layers[i](
                x=x_2,
                edge_index=edge_index_2,
                edge_weight=edge_weight,
                edge_type=edge_type,
            )

            # ---------------------------
            # Separate normalization per branch
            # ---------------------------
            x_1 = self.act(self.batch_norms_1[i](x_1))
            x_2 = self.act(self.batch_norms_2[i](x_2))

            # ---------------------------
            # Nodes -> VN update
            # ---------------------------
            vn_update = global_mean_pool(x_2, batch)
            vn_delta = self.act(self.vn_mlp[i](vn_update))

            # Residual VN update
            vn_state = vn_state + vn_delta

        # ---------------------------
        # 5. Graph-level fusion
        # ---------------------------
        node_repr = torch.cat([x_1, x_2], dim=-1)
        graph_repr = global_mean_pool(node_repr, batch)

        combined = torch.cat(
            [graph_repr, vn_state, esm_vec],
            dim=-1,
        )

        # ---------------------------
        # 6. Regression MLP
        # ---------------------------
        for layer in self.mlp:
            combined = layer(combined)

        return self.out(combined)


class GNN_Only_Predictor(torch.nn.Module):
    """
    GNN-only ablation model for protein-ligand binding affinity prediction.

    Uses:
        - Covalent-only graph branch
        - Full graph branch with covalent + intermolecular contact edges
        - Virtual node

    Does not use:
        - ESM embeddings
        - FiLM conditioning
    """

    def __init__(
        self,
        in_channels,
        num_gnn_layers,
        linear_out_channels,
        hidden_dim=128,
        dropout_gnn=0.1,
        dropout_mlp=0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers

        # ---------------------------
        # Input projection
        # ---------------------------
        self.input_transform = Linear(in_channels, self.hidden_dim)

        # ---------------------------
        # Shared GNN layers
        # ---------------------------
        self.gnn_layers = ModuleList()

        # Separate BatchNorms for the two branches.
        self.batch_norms_1 = ModuleList()
        self.batch_norms_2 = ModuleList()

        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                Gate_Augmented_GATv2_Layer(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    heads=4,
                    dropout=dropout_gnn,
                    save_attention=False,
                )
            )

            self.batch_norms_1.append(BatchNorm1d(self.hidden_dim))
            self.batch_norms_2.append(BatchNorm1d(self.hidden_dim))

        self.act = LeakyReLU(0.01)

        # ---------------------------
        # Virtual node
        # ---------------------------
        self.vn_embedding = Parameter(torch.zeros(1, self.hidden_dim))

        self.vn_mlp = ModuleList()
        self.vn_gate = ModuleList()
        self.vn_norm = ModuleList()

        for _ in range(num_gnn_layers):
            self.vn_mlp.append(
                Sequential(
                    Linear(self.hidden_dim, self.hidden_dim),
                    LeakyReLU(0.01),
                    Linear(self.hidden_dim, self.hidden_dim),
                )
            )

            self.vn_gate.append(Linear(self.hidden_dim * 2, 1))
            self.vn_norm.append(LayerNorm(self.hidden_dim))

        # ---------------------------
        # Final MLP
        # ---------------------------
        # graph_repr: x_1 + x_2 = 128 + 128
        # vn_state: 128
        #
        # Total = 128 * 2 + 128 = 384
        curr_dim = (self.hidden_dim * 2) + self.hidden_dim

        self.mlp = ModuleList()

        for out_dim in linear_out_channels:
            self.mlp.append(
                Sequential(
                    Linear(curr_dim, out_dim),
                    BatchNorm1d(out_dim),
                    LeakyReLU(0.01),
                    Dropout(dropout_mlp),
                )
            )
            curr_dim = out_dim

        self.out = Linear(curr_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset all trainable modules.
        """
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

        torch.nn.init.xavier_uniform_(self.vn_embedding)

    def forward(self, data):
        x = data.x
        edge_index_1 = data.edge_index_1
        edge_index_2 = data.edge_index_2
        edge_weight = data.edge_weight
        edge_type = data.edge_type
        batch = data.batch

        # ---------------------------
        # 1. Input transformation
        # ---------------------------
        x = self.input_transform(x)

        # ---------------------------
        # 2. Virtual-node initialization
        # ---------------------------
        num_graphs = int(batch.max().item()) + 1
        vn_state = self.vn_embedding.expand(num_graphs, -1)

        # ---------------------------
        # 3. Dual-path GNN
        # ---------------------------
        # x_1: covalent-only branch
        # x_2: full-graph branch
        x_1 = x
        x_2 = x

        for i in range(self.num_gnn_layers):
            # ---------------------------
            # VN -> nodes
            # ---------------------------
            vn_msg = vn_state.index_select(0, batch)

            # Use full-graph branch as context for VN gate
            z = torch.sigmoid(
                self.vn_gate[i](
                    torch.cat([x_2, vn_msg], dim=-1)
                )
            )

            x_1 = (1.0 - z) * x_1 + z * vn_msg
            x_2 = (1.0 - z) * x_2 + z * vn_msg

            # ---------------------------
            # Message passing
            # ---------------------------
            x_1 = self.gnn_layers[i](
                x=x_1,
                edge_index=edge_index_1,
            )

            x_2 = self.gnn_layers[i](
                x=x_2,
                edge_index=edge_index_2,
                edge_weight=edge_weight,
                edge_type=edge_type,
            )

            # ---------------------------
            # Separate normalization per branch
            # ---------------------------
            x_1 = self.act(self.batch_norms_1[i](x_1))
            x_2 = self.act(self.batch_norms_2[i](x_2))

            # ---------------------------
            # Nodes -> VN update
            # ---------------------------
            vn_update = global_mean_pool(x_2, batch)
            vn_delta = self.act(self.vn_mlp[i](vn_update))

            # Residual VN update
            vn_state = vn_state + vn_delta

        # ---------------------------
        # 4. Graph-level fusion
        # ---------------------------
        node_repr = torch.cat([x_1, x_2], dim=-1)
        graph_repr = global_mean_pool(node_repr, batch)

        combined = torch.cat(
            [graph_repr, vn_state],
            dim=-1,
        )

        # ---------------------------
        # 5. Regression MLP
        # ---------------------------
        for layer in self.mlp:
            combined = layer(combined)

        return self.out(combined)

def get_reuploading_vqc(n_qubits, qnn_layers):
        """
        Data re-uploading VQC used as a compact nonlinear regression head.

        Input:
            inputs: [batch_size, n_qubits]

        Output:
            expectation values: [batch_size, n_qubits]
        """

        def circuit(inputs, entangling_weights, embedding_weights):
            qml.AngleEmbedding(
                features=inputs,
                wires=range(n_qubits),
                rotation="Y",
            )

            for i in range(qnn_layers):
                qml.StronglyEntanglingLayers(
                    weights=entangling_weights[i],
                    wires=range(n_qubits),
                )

                qml.AngleEmbedding(
                    features=inputs * embedding_weights[i],
                    wires=range(n_qubits),
                    rotation="X",
                )

            qml.StronglyEntanglingLayers(
                weights=entangling_weights[-1],
                wires=range(n_qubits),
            )

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        weight_shapes = {
            "entangling_weights": (qnn_layers + 1, 1, n_qubits, 3),
            "embedding_weights": (qnn_layers, n_qubits),
        }

        return circuit, weight_shapes

class GNN_Only_VQC_Predictor(torch.nn.Module):
    """
    GNN-only encoder with a VQC regression head.

    Uses:
        - Covalent-only graph branch
        - Full graph branch with covalent + intermolecular contact edges
        - Virtual node
        - Low-dimensional projection to a VQC head

    Does not use:
        - ESM embeddings
        - FiLM conditioning

    This is a controlled head-level ablation:
        GNN encoder + classical MLP head
        vs
        GNN encoder + VQC head
    """

    def __init__(
        self,
        in_channels,
        num_gnn_layers,
        hidden_dim=128,
        dropout_gnn=0.1,
        dropout_mlp=0.3,
        n_qubits=8,
        qnn_layers=3,
        q_device="default.qubit",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.n_qubits = n_qubits
        self.qnn_layers = qnn_layers

        # ---------------------------
        # Input projection
        # ---------------------------
        self.input_transform = Linear(in_channels, self.hidden_dim)

        # ---------------------------
        # Shared GNN layers
        # ---------------------------
        self.gnn_layers = ModuleList()
        self.batch_norms_1 = ModuleList()
        self.batch_norms_2 = ModuleList()

        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                Gate_Augmented_GATv2_Layer(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    heads=4,
                    dropout=dropout_gnn,
                    save_attention=False,
                )
            )

            self.batch_norms_1.append(BatchNorm1d(self.hidden_dim))
            self.batch_norms_2.append(BatchNorm1d(self.hidden_dim))

        self.act = LeakyReLU(0.01)

        # ---------------------------
        # Virtual node
        # ---------------------------
        self.vn_embedding = Parameter(torch.zeros(1, self.hidden_dim))

        self.vn_mlp = ModuleList()
        self.vn_gate = ModuleList()
        self.vn_norm = ModuleList()

        for _ in range(num_gnn_layers):
            self.vn_mlp.append(
                Sequential(
                    Linear(self.hidden_dim, self.hidden_dim),
                    LeakyReLU(0.01),
                    Linear(self.hidden_dim, self.hidden_dim),
                )
            )

            self.vn_gate.append(Linear(self.hidden_dim * 2, 1))
            self.vn_norm.append(LayerNorm(self.hidden_dim))

        # ---------------------------
        # Graph representation size
        # ---------------------------
        # graph_repr: x_1 + x_2 = 128 + 128
        # vn_state: 128
        # Total = 384
        graph_dim = (self.hidden_dim * 2) + self.hidden_dim

        # ---------------------------
        # Classical bridge into quantum input
        # ---------------------------
        self.quantum_bridge = Sequential(
            Linear(graph_dim, 64),
            LayerNorm(64),
            torch.nn.PReLU(),
            Linear(64, n_qubits),
        )

        # ---------------------------
        # VQC layer
        # ---------------------------
        dev = qml.device(q_device, wires=n_qubits)
        vqc_func, weight_shapes = get_reuploading_vqc(
            n_qubits=n_qubits,
            qnn_layers=qnn_layers,
        )

        qnode = qml.QNode(
            vqc_func,
            dev,
            interface="torch",
            diff_method="backprop",
        )

        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # ---------------------------
        # Small classical readout
        # ---------------------------
        self.out = Sequential(
            Linear(n_qubits, 16),
            torch.nn.PReLU(),
            Linear(16, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset all trainable classical modules.
        PennyLane TorchLayer parameters are initialized internally.
        """
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                # TorchLayer does not always expose the same reset interface.
                if isinstance(module, qml.qnn.TorchLayer):
                    continue
                module.reset_parameters()

        torch.nn.init.xavier_uniform_(self.vn_embedding)

    def encode_graph(self, data):
        """
        Same encoder logic as GNN_Only_Predictor.
        Returns graph-level representation before the regression head.
        """
        x = data.x
        edge_index_1 = data.edge_index_1
        edge_index_2 = data.edge_index_2
        edge_weight = data.edge_weight
        edge_type = data.edge_type
        batch = data.batch

        x = self.input_transform(x)

        num_graphs = int(batch.max().item()) + 1
        vn_state = self.vn_embedding.expand(num_graphs, -1)

        x_1 = x
        x_2 = x

        for i in range(self.num_gnn_layers):
            vn_msg = vn_state.index_select(0, batch)

            z = torch.sigmoid(
                self.vn_gate[i](
                    torch.cat([x_2, vn_msg], dim=-1)
                )
            )

            x_1 = (1.0 - z) * x_1 + z * vn_msg
            x_2 = (1.0 - z) * x_2 + z * vn_msg

            x_1 = self.gnn_layers[i](
                x=x_1,
                edge_index=edge_index_1,
            )

            x_2 = self.gnn_layers[i](
                x=x_2,
                edge_index=edge_index_2,
                edge_weight=edge_weight,
                edge_type=edge_type,
            )

            x_1 = self.act(self.batch_norms_1[i](x_1))
            x_2 = self.act(self.batch_norms_2[i](x_2))

            vn_update = global_mean_pool(x_2, batch)
            vn_delta = self.act(self.vn_mlp[i](vn_update))

            vn_state = vn_state + vn_delta

        node_repr = torch.cat([x_1, x_2], dim=-1)
        graph_repr = global_mean_pool(node_repr, batch)

        combined = torch.cat(
            [graph_repr, vn_state],
            dim=-1,
        )

        return combined

    def forward(self, data):
        graph_embedding = self.encode_graph(data)
        q_in = self.quantum_bridge(graph_embedding)
        q_in = torch.tanh(q_in) * np.pi  # Normalize to [-π, π] for quantum gates
        
        # PennyLane default.qubit runs on CPU, but PyTorch autograd works through device transfers
        device = q_in.device
        q_out = self.qlayer(q_in.cpu()).to(device)
        
        return self.out(q_out)


class ESM_Only_Predictor(torch.nn.Module):
    """
    Ablation model using only the precomputed ESM embedding.

    No graph features, no ligand features, no virtual node,
    and no FiLM conditioning are used.
    """

    def __init__(
        self,
        linear_out_channels,
        esm_dim=320,
        dropout_mlp=0.3,
    ):
        super().__init__()

        curr_dim = esm_dim

        self.mlp = ModuleList()

        for out_dim in linear_out_channels:
            self.mlp.append(
                Sequential(
                    Linear(curr_dim, out_dim),
                    BatchNorm1d(out_dim),
                    LeakyReLU(0.01),
                    Dropout(dropout_mlp),
                )
            )
            curr_dim = out_dim

        self.out = Linear(curr_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset all trainable modules.
        """
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, data):
        esm_vec = data.esm_vec

        # Safety for a single unbatched graph
        if esm_vec.dim() == 1:
            esm_vec = esm_vec.unsqueeze(0)

        for layer in self.mlp:
            esm_vec = layer(esm_vec)

        return self.out(esm_vec)
    
    