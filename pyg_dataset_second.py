import base64
import hashlib
import json
import random
from pathlib import Path

import codetiming
import matplotlib
import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as pygUtils
import Utilities.utils as utils
import yaml
from matplotlib import pyplot as plt
from my_graphs_dataset import GraphDataset
from torch_geometric.data import InMemoryDataset, download_url, extract_zip


from torch_geometric.nn import GATConv
from torch.nn import functional as F

class Net(torch.nn.Module):
    def __init__(self, hidden_channels=256):
        super().__init__()
        self.conv1 = GATConv(8, hidden_channels, heads=4)
        self.lin1 = torch.nn.Linear(8, 4 * hidden_channels)
        self.conv2 = GATConv(4 * hidden_channels, hidden_channels, heads=4)
        self.lin2 = torch.nn.Linear(4 * hidden_channels, 4 * hidden_channels)
        self.conv3 = GATConv(4 * hidden_channels, hidden_channels, heads=4)
        self.lin3 = torch.nn.Linear(4 * hidden_channels, 4 * hidden_channels)
        self.conv5 = GATConv(4 * hidden_channels, hidden_channels, heads=4)
        self.lin5 = torch.nn.Linear(4 * hidden_channels, 4 * hidden_channels)
        self.conv4 = GATConv(4 * hidden_channels, 1, heads=6,
                             concat=False)
        self.lin4 = torch.nn.Linear(4 * hidden_channels, 1)

    def forward(self, x, edge_index):
        x = x.to(device='cuda:0')
        edge_index = edge_index.to(device='cuda:0')
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = F.elu(self.conv3(x, edge_index) + self.lin3(x))
        #x = F.elu(self.conv5(x, edge_index) + self.lin5(x))
        x = self.conv4(x, edge_index) + self.lin4(x)
        return x.squeeze()



class MIDSdataset(InMemoryDataset):
    def __init__(self, root, loader: GraphDataset, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.loader = loader

        super().__init__(root, transform, pre_transform, pre_filter)

        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return str(self.loader.raw_files_dir.resolve())

    @property
    def raw_file_names(self):
        """
        Return a list of all raw files in the dataset.

        This method has two jobs. The returned list with raw files is compared
        with the files currently in raw directory. Files that are missing are
        automatically downloaded using download method. The second job is to
        return the list of raw file names that will be used in the process
        method.
        """
        with open(Path(self.root) / "file_list.yaml", "r") as file:
            raw_file_list = sorted(yaml.safe_load(file))
        return raw_file_list

    @property
    def processed_file_names(self):
        """
        Return a list of all processed files in the dataset.

        If a processed file is missing, it will be automatically created using
        the process method.

        That means that if you want to reprocess the data, you need to delete
        the processed files and reimport the dataset.
        """
        dataset_props = json.dumps([self.loader.selection, self.features])
        sha256_hash = hashlib.sha256(dataset_props.encode("utf-8")).digest()
        hash_string = base64.urlsafe_b64encode(sha256_hash).decode("utf-8")[:10]

        return [f"data_{hash_string}.pt"]

    def download(self):
        """Automatically download raw files if missing."""
        # TODO: Should check and download only missing files.
        # zip_file = Path(self.root) / "raw_data.zip"
        # zip_file.unlink(missing_ok=True)  # Delete the exising zip file.
        # download_url(raw_download_url, self.root, filename="raw_data.zip")
        # extract_zip(str(zip_file.resolve()), self.raw_dir)
        raise NotImplementedError("Automatic download is not implemented yet.")

    def process(self):
        """Process the raw files into a graph dataset."""
        # Read data into huge `Data` list.
        data_list = []
        prob_model = torch.load('/home/jovyan/models/3-30_probability.pth')
        for graph in self.loader.graphs(batch_size=1):
            data_list.append(self.make_data(graph, prob_model))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        #data, slices = self.collate(data_list)
        self.save(data_list, self.processed_paths[0])

    # Define features in use.
    feature_functions = {
        # node features
        "degree": lambda g: {n: float(g.degree(n)) for n in g.nodes()},
        "degree_centrality": nx.degree_centrality,
        "random": lambda g: nx.random_layout(g, seed=np.random),
        "avg_neighbor_degree": nx.average_neighbor_degree,
        "closeness_centrality": nx.closeness_centrality,
        # graph features
        "number_of_nodes": lambda g: [nx.number_of_nodes(g)] * nx.number_of_nodes(g),
        "graph_density": lambda g: [nx.density(g)] * nx.number_of_nodes(g),
        #"probability": lambda g: {n: float(g.degree(n)) for n in g.nodes()}, # placeholder
    }

    def make_data(self, G, model):
        """Create a PyG data object from a graph object."""

        # Compute and add features to the nodes in the graph.
        for feature in self.feature_functions:
            feature_val = self.feature_functions[feature](G)
            for node in G.nodes():
                G.nodes[node][feature] = feature_val[node]

        torch_G = pygUtils.from_networkx(G, group_node_attrs=list(self.feature_functions.keys()))
        true_labels = MIDSdataset.get_labels(utils.find_MIDS(G), G.number_of_nodes())
        data = torch.zeros(len(true_labels[0]))
        count = 0
        #print(true_labels)
        for labels in true_labels:
            #data.append(torch_G.clone())
            data = torch.add(data, labels)
            count += 1

        probability = torch.div(data,count)
        # Compute and add features to the nodes in the graph.
        prob = model(torch_G.x, torch_G.edge_index).detach().to(device='cpu')

        self.feature_functions['probability'] = nx.average_neighbor_degree
        for i, node in enumerate(G.nodes()):
            G.nodes[node]["probability"] = prob[i] # for real probabilities 
            #G.nodes[node]["probability"] = probability[i] #torch.tensor(max(0, min(1, probability[i] + random.randrange(-30000, 30000, 1)*0.00001))) # for ideal probabilities (with noise)


        torch_G = pygUtils.from_networkx(G, group_node_attrs=list(self.feature_functions.keys()))
        true_labels = MIDSdataset.get_labels(utils.find_MIDS(G), G.number_of_nodes())
        data = torch.zeros(len(true_labels[0]))
        self.feature_functions.pop("probability")
        torch_G.y = torch.cat(true_labels) # all solutions

        return torch_G

    @property
    def features(self):
        return list(self.feature_functions.keys())

    @staticmethod
    def get_labels(mids, num_nodes):
        # Encode found cliques as support vectors.
        for i, nodes in enumerate(mids):
            mids[i] = torch.zeros(num_nodes)
            mids[i][nodes] = 1
        return mids

    @staticmethod
    def visualize_data(data):
        G = pygUtils.to_networkx(data, to_undirected=True)
        nx.draw(G, with_labels=True, node_color=torch.split(data.y, data.num_nodes)[0], cmap=matplotlib.colormaps["bwr"])
        plt.show()


def inspect_dataset(dataset, num_graphs=1):
    for i in random.sample(range(len(dataset)), num_graphs):
        data = dataset[i]  # Get a random graph object

        print()
        print(data)
        print("=============================================================")

        # Gather some statistics about the first graph.
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")

        MIDSdataset.visualize_data(data)


def main():
    root = Path(__file__).parent / "Dataset"
    selected_graph_sizes = {
                            #3: -1,
                            #4: -1,
                            #5: -1,
                            #6: -1,
                            #7: -1,
                            #8:  10000,
                            #9:  10000,
                            #10: 10000,
                            #11: 10000,
                            #12: 10000,
                            #13: 10000,
                            #14: 10000,
                            #15: 10000,
                            #20: 10000,
                            50: 2000,
                        }
    loader = GraphDataset(selection=selected_graph_sizes)

    with codetiming.Timer():
        dataset = MIDSdataset(root, loader)

    print()
    print(f"Dataset: {dataset}:")
    print("====================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")

    inspect_dataset(dataset, num_graphs=1)


if __name__ == "__main__":
    main()
