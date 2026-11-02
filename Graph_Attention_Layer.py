import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter, LeakyReLU, Sequential, Sigmoid
from torch_geometric.nn import GATv2Conv


class Gate_Augmented_GATv2_Layer(torch.nn.Module):
    """
    Gate-augmented GATv2 layer with residual connection and optional edge attributes.

    This layer supports two types of edges:
        edge_type = 0: covalent edge
        edge_type = 1: intermolecular/contact edge

    Edge attributes are constructed from:
        - learned RBF-like distance weight
        - covalent edge indicator
        - contact edge indicator
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=4,
        add_self_loops=True,
        dropout=0.1,
        save_attention=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.save_attention = save_attention

        if in_channels != out_channels:
            self.res_linear = Linear(in_channels, out_channels)
        else:
            self.res_linear = None

        # edge_dim=3 because we use:
        # [rbf_distance_weight, is_covalent, is_contact]
        self.gat_conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=False,
            edge_dim=3,
            add_self_loops=add_self_loops,
        )

        # Learnable parameters for the distance-based RBF weighting
        self.alpha_rbf = Parameter(torch.tensor(3.0))
        self.beta_rbf = Parameter(torch.tensor(1.0))

        self.linear_gate = Sequential(
            Linear(in_channels + out_channels, out_channels),
            LeakyReLU(0.01),
            Linear(out_channels, out_channels),
            Sigmoid(),
        )

        self.act = LeakyReLU(0.01)
        self.dropout = torch.nn.Dropout(dropout)

        self.attn_weights = None

    def build_edge_attr(self, edge_weight, edge_type):
        """
        Build edge attributes for GATv2.

        Parameters
        ----------
        edge_weight : Tensor, shape [num_edges]
            Edge distances or weights.
            For covalent edges, this can be 1.0.
            For contact edges, this should be the actual distance.

        edge_type : Tensor, shape [num_edges]
            0 for covalent edges, 1 for contact/intermolecular edges.

        Returns
        -------
        edge_attr : Tensor, shape [num_edges, 3]
            Edge attributes:
                [rbf_weight, is_covalent, is_contact]
        """
        edge_weight = edge_weight.view(-1)
        edge_type = edge_type.view(-1).long()

        dist = torch.clamp(edge_weight, min=0.0, max=10.0)

        beta = F.softplus(self.beta_rbf) + 1e-6
        rbf_weight = torch.exp(-((dist - self.alpha_rbf) ** 2) / beta)

        is_covalent = (edge_type == 0).float()
        is_contact = (edge_type == 1).float()

        # For covalent edges, edge_weight=1.0 is only a placeholder,
        # not a real physical distance. So we force the distance component
        # to 1.0 and let the indicator encode that this is a covalent edge.
        rbf_weight = torch.where(
            is_covalent.bool(),
            torch.ones_like(rbf_weight),
            rbf_weight,
        )

        edge_attr = torch.stack(
            [rbf_weight, is_covalent, is_contact],
            dim=-1,
        )

        return edge_attr

    def forward(self, x, edge_index, edge_weight=None, edge_type=None):
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Node features.

        edge_index : Tensor
            Graph connectivity.

        edge_weight : Tensor or None
            Edge weights/distances.

        edge_type : Tensor or None
            Edge type labels:
                0 = covalent
                1 = contact

        Returns
        -------
        Tensor
            Updated node representations.
        """
        if edge_weight is not None:
            if edge_type is None:
                raise ValueError(
                    "edge_type must be provided when edge_weight is provided. "
                    "Use edge_type=0 for covalent edges and edge_type=1 for contact edges."
                )

            edge_attr = self.build_edge_attr(edge_weight, edge_type)
        else:
            edge_attr = None

        # Important:
        # Safely extract tensor regardless of PyG version
        if self.save_attention:
            out, (edge_index_attn, alpha) = self.gat_conv(
                x,
                edge_index,
                edge_attr=edge_attr,
                return_attention_weights=True,
            )

            self.attn_weights = (
                edge_index_attn.detach().cpu(),
                alpha.detach().cpu(),
            )
        else:
            result = self.gat_conv(
                x,
                edge_index,
                edge_attr=edge_attr,
            )
            out = result[0] if isinstance(result, tuple) else result

        x_out = self.dropout(self.act(out))

        x_residual = x if self.res_linear is None else self.res_linear(x)

        gate_input = torch.cat([x, x_out], dim=-1)
        z = self.linear_gate(gate_input)

        return (1.0 - z) * x_residual + z * x_out

    def reset_parameters(self):
        self.gat_conv.reset_parameters()

        if self.res_linear is not None:
            self.res_linear.reset_parameters()

        for module in self.linear_gate:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        torch.nn.init.constant_(self.alpha_rbf, 3.0)
        torch.nn.init.constant_(self.beta_rbf, 1.0)

        self.attn_weights = None