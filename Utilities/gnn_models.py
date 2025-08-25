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


# TODO: input checking
# NOTE: hyperparameters are fixed to those found by the non-end-to-end training
class CombinedGNN(torch.nn.Module):
    def __init__(self, architecture, in_channels, num_layers, out_channels=1, **kwargs):
        super().__init__()

        kwargs1 = kwargs.copy()
        kwargs2 = kwargs.copy()
        hidden_channels = [0, 0]

        arch1, arch2 = architecture.split("+")

        if arch1 == "GATLinNet":
            kwargs1["act"] = "relu"
            kwargs1.pop("jk")
            hidden_channels[0] = 64
        elif arch1 == "MLP":
            kwargs1["act"] = "relu"
            hidden_channels[0] = 128
        elif arch1 == "GCN":
            kwargs1["act"] = "relu"
            hidden_channels[0] = 64
        elif arch1 == "GIN":
            kwargs1["act"] = "tanh"
            hidden_channels[0] = 32
            kwargs1["train_eps"] = True
        elif arch1 == "GraphSAGE":
            kwargs1["act"] = "relu"
            hidden_channels[0] = 32
        elif arch1 == "GAT":
            kwargs1["act"] = "relu"
            hidden_channels[0] = 128
            kwargs1["v2"] = True

        if arch2 == "GATLinNet":
            kwargs2["act"] = "tanh"
            kwargs2.pop("jk")
            hidden_channels[1] = 16
        elif arch2 == "MLP":
            kwargs2["act"] = "relu"
            hidden_channels[1] = 128
        elif arch2 == "GCN":
            kwargs2["act"] = "relu"
            hidden_channels[1] = 128
        elif arch2 == "GIN":
            kwargs2["act"] = "tanh"
            hidden_channels[1] = 32
            kwargs2["train_eps"] = True
        elif arch2 == "GraphSAGE":
            kwargs2["act"] = "elu"
            hidden_channels[1] = 32
        elif arch2 == "GAT":
            kwargs2["act"] = "tanh"
            hidden_channels[1] = 16
            kwargs2["v2"] = True


        if arch1 == "GATLinNet":
            self.gnn1 = GATLinNet(in_channels, hidden_channels[0], num_layers, 9, **kwargs1)
        else:
            self.gnn1 = GNNWrapper(premade_gnns[arch1], in_channels, hidden_channels[0], num_layers, 9, **kwargs1)

        if arch2 == "GATLinNet":
            self.gnn2 = GATLinNet(9, hidden_channels[1], num_layers, out_channels, **kwargs2)
        else:
            self.gnn2 = GNNWrapper(premade_gnns[arch2], 9, hidden_channels[1], num_layers, out_channels, **kwargs2)


    def forward(self, x, edge_index, batch=None):
        x = self.gnn1(x, edge_index, batch)
        x = self.gnn2(x, edge_index, batch)
        return x

premade_gnns = {x.__name__: x for x in [MLP, GCN, GraphSAGE, GIN, GAT, PNA]}
custom_gnns = {x.__name__: x for x in [GATLinNet]}
