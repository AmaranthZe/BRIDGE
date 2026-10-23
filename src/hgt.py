"""
Heterogeneous Graph Transformer (HGT) Implementation.

Uses PyTorch Geometric's HGTConv for memory efficiency.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


class HGT(nn.Module):
    """
    Heterogeneous Graph Transformer using PyG's implementation.
    """

    def __init__(
        self,
        node_counts: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        num_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.3,
        edge_bias_kappa: float = 1.0,
        share_go_weights: bool = True,
        share_gene_weights: bool = False,
    ):
        super().__init__()

        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Learnable node embeddings
        self.embeddings = nn.ModuleDict()
        for node_type in self.node_types:
            self.embeddings[node_type] = nn.Embedding(
                node_counts[node_type],
                hidden_channels
            )

        # Try to use PyG's HGTConv
        try:
            from torch_geometric.nn import HGTConv
            self.use_pyg_hgt = True

            self.layers = nn.ModuleList()
            for i in range(num_layers):
                conv = HGTConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=heads,
                )
                self.layers.append(conv)
        except ImportError:
            self.use_pyg_hgt = False
            logger.warning("PyG HGTConv not available, using simple message passing")
            # Fallback to simple linear layers
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                layer = nn.ModuleDict()
                for nt in self.node_types:
                    layer[nt] = nn.Linear(hidden_channels, hidden_channels)
                self.layers.append(layer)

        # Output projection
        self.out_proj = nn.ModuleDict()
        for node_type in self.node_types:
            self.out_proj[node_type] = nn.Linear(hidden_channels, out_channels)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)

    def forward(
        self,
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], Tensor]] = None,
        node_indices: Optional[Dict[str, Tensor]] = None,
        use_checkpointing: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Forward pass with optional gradient checkpointing.
        """
        # Get initial embeddings
        x_dict = {
            nt: self.embeddings[nt].weight
            for nt in self.node_types
        }

        if self.use_pyg_hgt:
            # Use PyG's HGTConv with gradient checkpointing
            for layer in self.layers:
                if use_checkpointing and self.training:
                    # Checkpoint to save memory during backward
                    x_dict = checkpoint(
                        self._layer_forward,
                        x_dict, edge_index_dict, layer,
                        use_reentrant=False,
                    )
                else:
                    x_dict = self._layer_forward(x_dict, edge_index_dict, layer)
        else:
            # Simple message passing fallback
            for layer in self.layers:
                new_x_dict = {}
                for nt in self.node_types:
                    new_x_dict[nt] = self.dropout(F.relu(layer[nt](x_dict[nt])))
                x_dict = new_x_dict

        # Output projection
        out_dict = {}
        for node_type, x in x_dict.items():
            out_dict[node_type] = self.out_proj[node_type](x)

        return out_dict

    def _layer_forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        layer,
    ) -> Dict[str, Tensor]:
        """Single layer forward pass for checkpointing."""
        x_dict = layer(x_dict, edge_index_dict)
        x_dict = {k: self.dropout(F.relu(v)) for k, v in x_dict.items()}
        return x_dict

    def get_all_embeddings(
        self,
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], Tensor]] = None,
    ) -> Dict[str, Tensor]:
        return self.forward(edge_index_dict, edge_attr_dict)

    def get_gene_embeddings(
        self,
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        out_dict = self.forward(edge_index_dict, edge_attr_dict)
        return out_dict['Human_Gene'], out_dict['Mouse_Gene']
