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
import Utilities.mids_utils as mids_utils
import yaml
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from my_graphs_dataset import GraphDataset
from torch.nn import functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import GATConv


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
        self.conv4 = GATConv(4 * hidden_channels, 1, heads=6, concat=False)
        self.lin4 = torch.nn.Linear(4 * hidden_channels, 1)

    def forward(self, x, edge_index):
        x = x.to(device="cuda:0")
        edge_index = edge_index.to(device="cuda:0")
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = F.elu(self.conv3(x, edge_index) + self.lin3(x))
        # x = F.elu(self.conv5(x, edge_index) + self.lin5(x))
        x = self.conv4(x, edge_index) + self.lin4(x)
        return x.squeeze()


class MIDSDataset(InMemoryDataset):
    def __init__(self, root, loader: GraphDataset, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        if loader is None:
            loader = GraphDataset()
        self.loader = loader

        print("*****************************************")
        print(f"** Creating dataset with ID {self.hash_representation} **")
        print("*****************************************")

        super().__init__(root, transform, pre_transform, pre_filter)

        self.load(self.processed_paths[0])

        if selected_features := kwargs.get("selected_features"):
            self.filter_features(selected_features)

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
        return [f"data_{self.hash_representation}.pt"]

    @property
    def hash_representation(self):
        dataset_props = json.dumps(
            [
                self.__class__.__name__,
                self.loader.hashable_selection,
                self.features,
                self.target_function.__name__,
                self.loader.seed,
            ]
        )
        sha256_hash = hashlib.sha256(dataset_props.encode("utf-8")).digest()
        hash_string = base64.urlsafe_b64encode(sha256_hash).decode("utf-8")[:10]
        return hash_string

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
        for graph in self.loader.graphs(batch_size=1, raw=False):
            data_list.append(self.make_data(graph))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # data, slices = self.collate(data_list)
        self.save(data_list, self.processed_paths[0])

    # *************************
    # *** Feature functions ***
    # *************************
    @staticmethod
    def true_mids_probabilities(G):
        features = {}
        probs = MIDSDataset.true_probabilities(G)
        for i, node in enumerate(G.nodes()):
            features[node] = probs[i].item()

    @staticmethod
    def noisy_mids_probabilities(G):
        features = {}
        probs = MIDSDataset.true_probabilities(G)
        for i, node in enumerate(G.nodes()):
            features[node] = max(0, min(1, probs[i].item() + random.uniform(-0.3, 0.3)))

    @staticmethod
    def predicted_mids_probabilities(G, torch_G, model):
        prob = model(torch_G.x, torch_G.edge_index).detach().to(device="cpu")
        for i, node in enumerate(G.nodes()):
            G.nodes[node]["predicted_probability"] = prob[i].item()
        return G

    # ************************

    # ************************
    # *** Target functions ***
    # ************************
    @staticmethod
    def empty_function(G):
        return 0

    @staticmethod
    def true_probabilities(G):
        true_labels = MIDSDataset.get_labels(mids_utils.find_MIDS(G), G.number_of_nodes())
        data = torch.zeros(len(true_labels[0]))
        count = 0
        for labels in true_labels:
            data = torch.add(data, labels)
            count += 1

        return torch.div(data, count)

    @staticmethod
    def true_labels_single(G):
        true_labels = MIDSDataset.get_labels(mids_utils.find_MIDS(G), G.number_of_nodes())
        return true_labels[random.randrange(0, len(true_labels))]

    @staticmethod
    def true_labels_all_padded(G):
        max_solutions = 10
        true_labels = MIDSDataset.get_labels(mids_utils.find_MIDS(G), G.number_of_nodes())
        true_labels = true_labels[:max_solutions]
        padding = [-1 * torch.ones(len(true_labels[0])) for _ in range(len(true_labels), max_solutions)]
        return torch.stack(true_labels+padding, dim=1)

    @staticmethod
    def true_labels_all_stacked(G):
        true_labels = MIDSDataset.get_labels(mids_utils.find_MIDS(G), G.number_of_nodes())
        return torch.cat(true_labels)
    # ************************

    # ******************************************
    # ******* Features and labels in use *******
    # This should be overridden in the subclass.
    # ******************************************
    feature_functions = {}
    target_function = empty_function
    probability_predictor = Net()

    def make_data(self, G):
        """Create a PyG data object from a graph object."""
        # Compute and add features to the nodes in the graph.
        for feature in self.feature_functions:
            if feature == "predicted_probability":
                continue
            feature_val = self.feature_functions[feature](G)
            for node in G.nodes():
                G.nodes[node][feature] = feature_val[node]

        initial_features = set(self.features) - {"predicted_probability"}
        torch_G = pygUtils.from_networkx(G, group_node_attrs=list(initial_features))

        if "predicted_probability" in self.feature_functions:
            G = self.predicted_mids_probabilities(G, torch_G, self.probability_predictor)
            torch_G = pygUtils.from_networkx(G, group_node_attrs=self.features)

        torch_G.y = self.target_function(G)

        return torch_G

    @property
    def features(self):
        return list(self.feature_functions.keys())

    def filter_features(self, selected_features):
        """Filter out features that are not in the selected features."""
        mask = np.array([name in selected_features for name in self.features])
        # FIXME: This is not a proper way, but I don't know what else to do.
        # https://github.com/pyg-team/pytorch_geometric/discussions/7684
        # This works only because it is applied to the whole dataset, and before
        # the split. After splitting, `data` and `_data` still hold references
        # to the whole dataset, so we can't modify only one part.
        assert self._data is not None
        self._data.x = self._data.x[:, mask]

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
        nx.draw(
            G,
            with_labels=True,
            node_color=torch.split(data.y, data.num_nodes)[0],
            cmap=matplotlib.colormaps["viridis"],
            vmin=0,
            vmax=1,
        )
        # Add a sidebar with the color map
        sm = ScalarMappable(cmap=matplotlib.colormaps["viridis"], norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca())
        plt.show()


class MIDSProbabilitiesDataset(MIDSDataset):
    def __init__(self, root, loader: GraphDataset, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        super().__init__(root, loader, transform, pre_transform, pre_filter, **kwargs)

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
    }

    target_function = staticmethod(MIDSDataset.true_probabilities)


class MIDSLabelsDataset(MIDSDataset):
    def __init__(self, root, loader: GraphDataset, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        super().__init__(root, loader, transform, pre_transform, pre_filter, **kwargs)

    # probability_predictor = torch.load('/home/jovyan/models/3-30_probability.pth')
    probability_predictor = Net().to(device="cuda:0")

    feature_functions = {
        # node features
        "degree": lambda g: {n: float(g.degree(n)) for n in g.nodes()},
        "degree_centrality": nx.degree_centrality,
        "random": lambda g: nx.random_layout(g, seed=np.random),
        "avg_neighbor_degree": nx.average_neighbor_degree,
        "closeness_centrality": nx.closeness_centrality,
        # "predicted_probability": None,
        # graph features
        "number_of_nodes": lambda g: [nx.number_of_nodes(g)] * nx.number_of_nodes(g),
        "graph_density": lambda g: [nx.density(g)] * nx.number_of_nodes(g),
    }

    target_function = staticmethod(MIDSDataset.true_labels_single)


def inspect_dataset(dataset):
    if isinstance(dataset, InMemoryDataset):
        dataset_name = dataset.__repr__()
        # y_values = dataset.y
        y_name = dataset.target_function.__name__
        num_features = dataset.num_features
        features = dataset.features
    else:
        dataset_name = "N/A"
        # y_values = torch.tensor([data.y for data in dataset])
        y_name = "N/A"
        num_features = dataset[0].x.shape[1]
        features = "N/A"

    print()
    header = f"Dataset: {dataset_name}"
    print(header)
    print("=" * len(header))
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {num_features} ({features})")
    print(f"Target: {y_name}")
    # print(f"    Min: {y_values.min().item():.3f}")
    # print(f"    Max: {y_values.max().item():.3f}")
    # print(f"    Mean: {y_values.mean().item():.3f}")
    # print(f"    Std: {y_values.std().item():.3f}")
    print("=" * len(header))
    print()


def inspect_graphs(dataset, num_graphs=1):
    try:
        y_name = dataset.target_function(None)
    except AttributeError:
        y_name = "Target value"

    for i in random.sample(range(len(dataset)), num_graphs):
        data = dataset[i]  # Get a random graph object

        print()
        header = f"{i} - {data}"
        print(header)
        print("=" * len(header))

        # Gather some statistics about the graph.
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"{y_name}: {data.y}")
        print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")
        print(f"Features:\n{data.x}")
        print("=" * len(header))
        print()

        # MIDSDataset.visualize_data(data)


def main():
    root = Path(__file__).parent / "Dataset"
    selected_graph_sizes = {
        3: -1,
        # 4: -1,
        # 5: -1,
        # 6: -1,
        # 7: -1,
        # 8: -1,
        # 9:  10000,
        # 10: 10000,
        # 11: 10000,
        # 12: 10000,
        # 13: 10000,
        # 14: 10000,
        # 15: 10000,
        # 20: 5000,
        # 30: 5000,
    }
    loader = GraphDataset(selection=selected_graph_sizes)

    with codetiming.Timer():
        dataset = MIDSLabelsDataset(root, loader)

    inspect_dataset(dataset)
    inspect_graphs(dataset, num_graphs=2)


if __name__ == "__main__":
    main()
