import torch
from torch import nn, Tensor as T
import torch_geometric
import torch_geometric.nn as geom_nn


class GNN(nn.Module):
    def __init__(self, 
                 c_in: int, #Dimension of input features.
                 c_hidden: int, #Dimension of hidden features.
                 c_out: int, #Dimension of the output features.
                 num_layers: int, #Number of "hidden" graph layers.
                 layer_name: str, #String of the graph layer to use.
                 dp_rate: float = 0.1, #Dropout rate to apply throughout the network.
                 **kwargs, #Additional arguments for the graph layer (e.g. number of heads for GAT).
        ):
        super().__init__()
        gnn_layer_by_name = {
            "gcn": geom_nn.GCNConv, #https://arxiv.org/abs/1609.02907 (2016).
            "graphsage": geom_nn.SAGEConv, #https://arxiv.org/abs/1706.02216 (2017)
            "gat": geom_nn.GATConv, #https://arxiv.org/abs/1710.10903 (2017).
            "k-gnn": geom_nn.GraphConv, #https://arxiv.org/abs/1810.02244 (2018)
            "gatv2": geom_nn.GATv2Conv, #GAT from https://arxiv.org/abs/2105.14491 (2021).
        }
        layer_name = layer_name.lower()
        assert layer_name in gnn_layer_by_name.keys(), f"ERROR: Unknown graph layer name {layer_name}."
        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        Returns:
            Updated features per node.
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input. All PyTorch Geometric graph layer inherit the class "MessagePassing", hence we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x
