import torch
from torch_geometric.nn import (
    GAT,
    GCN,
    GIN,
    MLP,
    PNA,
    GraphSAGE,
    GATConv
)
from torch_geometric.nn.resolver import activation_resolver

class GNNWrapper(torch.nn.Module):
    def __init__(self, gnn_model, in_channels, hidden_channels, num_layers, out_channels=1, **kwargs):
        super().__init__()
        self.gnn = gnn_model(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, **kwargs)
        self.is_mlp = isinstance(self.gnn, MLP)

    def forward(self, x, edge_index, batch=None):
        if self.is_mlp:
            x = self.gnn(x=x, batch=batch)
        else:
            x = self.gnn(x=x, edge_index=edge_index, batch=batch)
        return x


class GATLinNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels=1, **kwargs):
        super().__init__()

        heads = kwargs.get("heads", 2)
        self.num_layers = num_layers
        self.act = activation_resolver(kwargs.get("act", "relu"))
        if kwargs.get("jk", None) is not None:
            raise ValueError("Jumping knowledge is not supported for this model.")

        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        if num_layers > 1:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
            self.lins.append(torch.nn.Linear(in_channels, heads * hidden_channels))
            in_channels = heads * hidden_channels

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
            self.lins.append(torch.nn.Linear(in_channels, heads * hidden_channels))
            in_channels = heads * hidden_channels

        self.convs.append(GATConv(in_channels, out_channels, heads=heads, concat=False))
        self.lins.append(torch.nn.Linear(in_channels, out_channels))

    def forward(self, x, edge_index, batch=None):
        for i in range(self.num_layers - 1):
            x = self.act(self.convs[i](x, edge_index) + self.lins[i](x))
        x = self.convs[-1](x, edge_index) + self.lins[-1](x)
        return x


premade_gnns = {x.__name__: x for x in [MLP, GCN, GraphSAGE, GIN, GAT, PNA]}
custom_gnns = {x.__name__: x for x in [GATLinNet]}