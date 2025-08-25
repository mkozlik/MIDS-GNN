import base64
import hashlib
import json
import random
import functools
from pathlib import Path

import codetiming
import matplotlib
import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as tg_utils
import torch_geometric.transforms as tg_transforms
import yaml
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from torch_geometric.data import InMemoryDataset, Data

import Utilities.mids_utils as mids_utils
from Utilities.gnn_models import GNNWrapper, GATLinNet
from my_graphs_dataset import GraphDataset


BEST_MODEL_PATH = Path(__file__).parents[0] / "Models"


class FeatureFilterTransform(tg_transforms.BaseTransform):
    """
    A transform that filters features of a graph's node attributes based on a given mask.

    Args:
        feature_index_mask (list or np.ndarray): A boolean mask or list of indices to filter the features.
    Methods:
        forward(data: Data) -> Data:
            Applies the feature filter to the node attributes of the input data object.
        __repr__() -> str:
            Returns a string representation of the transform with the mask.
    Example:
        >>> transform = FeatureFilterTransform([0, 2, 4])
        >>> data = Data(x=torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
        >>> transformed_data = transform(data)
        >>> print(transformed_data.x)
        tensor([[ 1,  3,  5],
                [ 6,  8, 10]])
    """
    # NOTE: This is a better way than doing self._data.x = self._data.x[:, mask]
    # See https://github.com/pyg-team/pytorch_geometric/discussions/7684.
    # This transform function will be automatically applied to each data object
    # when it is accessed. It might be a bit slower, but tensor slicing
    # shouldn't affect the performance too much. It is also following the
    # intended Dataset design.
    def __init__(self, feature_index_mask):
        self.mask = feature_index_mask

    def forward(self, data: Data) -> Data:
        if self.mask is not None:
            assert data.x is not None
            data.x = data.x[:, self.mask]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.mask})'


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

        selected_features = kwargs.get("selected_features")
        selected_extra_feature = kwargs.get("selected_extra_feature")
        mask = self.get_features_mask(selected_features, selected_extra_feature)

        # Add a feature filter transform to the transform pipeline, if needed.
        if not np.all(mask):
            feature_filter = FeatureFilterTransform(mask)
            if self.transform is not None:
                self.transform = tg_transforms.Compose([self.transform, feature_filter])
            else:
                self.transform = feature_filter

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
        return features

    @staticmethod
    def noisy_mids_probabilities(G):
        features = {}
        probs = MIDSDataset.true_probabilities(G)
        for i, node in enumerate(G.nodes()):
            features[node] = max(0, min(1, probs[i].item() + random.gauss(-0.0150, 0.1375)))
        return features

    @staticmethod
    def predicted_mids_probabilities(G, torch_G, model):
        features = {}
        # torch_G = torch_G.to(device="cuda")
        with torch.no_grad():
            prob = model(torch_G.x, torch_G.edge_index)#.detach()#.to(device="cpu")
        for i, node in enumerate(G.nodes()):
            features[node] = prob[i].item()
        return features

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
    extra_feature_functions = {}
    target_function = empty_function

    def make_data(self, G):
        """Create a PyG data object from a graph object."""
        # Compute and add features to the nodes in the graph.
        for feature in self.feature_functions:
            feature_val = self.feature_functions[feature](G)
            for node in G.nodes():
                G.nodes[node][feature] = feature_val[node]

        torch_G = tg_utils.from_networkx(G, group_node_attrs=self.features)

        # Extra probability features.
        for feature in self.extra_feature_functions:
            if feature == "predicted_probability":
                feature_val = self.extra_feature_functions[feature](G, torch_G, self.probability_predictor)
            else:
                feature_val = self.extra_feature_functions[feature](G)
            for node in G.nodes():
                G.nodes[node][feature] = feature_val[node]

        if self.extra_feature_functions:
            torch_G = tg_utils.from_networkx(G, group_node_attrs=self.features + self.extra_features)

        torch_G.y = self.target_function(G)

        return torch_G

    @property
    def features(self):
        return list(self.feature_functions.keys())

    @property
    def extra_features(self):
        return list(self.extra_feature_functions.keys())

    @functools.cached_property
    def feature_dims(self):
        feature_dims = {}
        G = tg_utils.to_networkx(self[0], to_undirected=True)
        for feature in self.feature_functions:
            feature_val = self.feature_functions[feature](G)
            try:
                feature_dims[feature] = len(feature_val[0])
            except TypeError:
                feature_dims[feature] = 1
        return feature_dims

    def get_features_mask(self, selected_features, selected_extra_feature):
        """Filter out features that are not in the selected features."""
        flags = []
        for feature in self.features:
            val = selected_features is None or feature in selected_features
            flags.extend([val] * self.feature_dims[feature])
        for feature in self.extra_features:
            val = selected_extra_feature is None or feature == selected_extra_feature
            flags.append(val)
        return np.array(flags)

    @staticmethod
    def get_labels(mids, num_nodes):
        # Encode found cliques as support vectors.
        for i, nodes in enumerate(mids):
            mids[i] = torch.zeros(num_nodes)
            mids[i][nodes] = 1
        return mids

    @staticmethod
    def visualize_data(data):
        G = tg_utils.to_networkx(data, to_undirected=True)
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
        "random": lambda g: nx.random_layout(g, seed=np.random), # This works because GraphDataset loader sets the seed
        "avg_neighbor_degree": nx.average_neighbor_degree,
        "closeness_centrality": nx.closeness_centrality,
        # graph features
        "number_of_nodes": lambda g: [nx.number_of_nodes(g)] * nx.number_of_nodes(g),
        "graph_density": lambda g: [nx.density(g)] * nx.number_of_nodes(g),
    }
    extra_feature_functions = {}

    target_function = staticmethod(MIDSDataset.true_probabilities)


class MIDSLabelsDataset(MIDSDataset):
    def __init__(self, root, loader: GraphDataset, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.probability_predictor = torch.load(BEST_MODEL_PATH / 'prob_model_best.pth')
        self.probability_predictor.to(device="cpu")
        self.probability_predictor.eval()

        self.source_dataset = kwargs.get("source_dataset")
        if override_target := kwargs.get("override_target_function"):
            self.target_function = override_target

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

    extra_feature_functions = {
        "predicted_probability": staticmethod(MIDSDataset.predicted_mids_probabilities),
        "true_probability": staticmethod(MIDSDataset.true_probabilities),
        "noisy_probability": staticmethod(MIDSDataset.noisy_mids_probabilities),
    }

    target_function = staticmethod(MIDSDataset.true_labels_all_padded)


    def process(self):
        if self.source_dataset is None:
            super().process()
        else:
            self.copy_process()

    def copy_process(self):
        """A private process override used for converting between lables for single and mulitple solutions."""
        assert self.source_dataset is not None

        data_list = []

        for data in self.source_dataset:
            if self.source_dataset.target_function == MIDSDataset.true_labels_all_padded and self.target_function == MIDSDataset.true_labels_single:
                data_list.append(self.filter_single_solution(data))
            else:
                raise NotImplementedError("Any conversion between labels other than multi->single is not implemented.")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # data, slices = self.collate(data_list)
        self.save(data_list, self.processed_paths[0])

    @staticmethod
    def filter_single_solution(data):
        # Invalid, padded solutions all contain -1 values in the column. We want to select a random valid solution.
        valid_solutions = data.y[0, :] != -1
        valid_solutions = data.y[:, valid_solutions]
        selected_solution = valid_solutions[:, random.randrange(0, valid_solutions.shape[1])]
        data.y = selected_solution
        return data


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


def inspect_graphs(dataset, graphs:int | list=1):
    try:
        y_name = dataset.target_function.__name__
    except AttributeError:
        y_name = "Target value"

    if isinstance(graphs, int):
        graphs = random.sample(range(len(dataset)), graphs)

    for i in graphs:
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


def analyze_labels_dataset(dataset):
    import pandas as pd
    from Utilities.graph_utils import graphs_to_tuple

    # Take individual graphs and store their properties in a DataFrame.
    nx_graphs = nx_graphs = [tg_utils.to_networkx(d, to_undirected=True) for d in dataset]
    graphs, node_nums, edge_nums = zip(*graphs_to_tuple(nx_graphs))

    # Extract the ground truth and predicted values from the dataset.
    ground_truth = [np.round(data.x.numpy(), 3)[:, 9] for data in dataset]
    predictions = [np.round(data.x.numpy(), 3)[:, 8] for data in dataset]

    df = pd.DataFrame(
        {
            "Graph": graphs,
            "Nodes": node_nums,
            "Edges": edge_nums,
            "True": ground_truth,
            "Predicted": predictions,
        }
    )

    df = df.sort_values(by="Nodes")
    print("\nDetailed results:")
    print("==================")
    print(df.head(10))

    # Calculate the average error and standard deviation over all nodes.
    errors = np.concatenate(
        [(np.array(row["Predicted"]) - np.array(row["True"])).squeeze() for _, row in df.iterrows()]
    )

    print("\nStatistics over all nodes:")
    print("===========================")
    print(f"Average error: {np.mean(errors):.3f}\n"
          f"Average absolute error: {np.mean(np.abs(errors)):.3f}\n"
          f"Standard deviation: {np.std(errors):.3f}")


def main():
    root = Path(__file__).parent / "Dataset"
    selected_graph_sizes = {
        "26-50_mix_100": -1,
        # "03-25_mix_750": -1,
    }
    loader = GraphDataset(selection=selected_graph_sizes, seed=42)

    with codetiming.Timer():
        dataset = MIDSLabelsDataset(root, loader, selected_extra_feature=None)
        # new_dataset = MIDSLabelsDataset(root, loader, selected_extra_feature=None, override_target_function=MIDSDataset.true_labels_single, source_dataset=dataset)
        # loaded_dataset = MIDSLabelsDataset(root, loader, selected_extra_feature=None, override_target_function=MIDSDataset.true_labels_single)


    inspect_dataset(dataset)
    # inspect_graphs(dataset, graphs=[0, 1])
    analyze_labels_dataset(dataset)

    # inspect_graphs(new_dataset, graphs=[0, 1])

    # inspect_graphs(loaded_dataset, graphs=[0, 1])


if __name__ == "__main__":
    main()
