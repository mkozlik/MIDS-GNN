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
        for graph in self.loader.graphs(batch_size=1):
            data_list.append(self.make_data(graph))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        #data, slices = self.collate(data_list)
        self.save(data_list, self.processed_paths[0])

    # Define features in use.
    feature_functions = {
        "degree": lambda g: {n: float(g.degree(n)) for n in g.nodes()},
        "degree_centrality": nx.degree_centrality,
        "random": lambda g: nx.random_layout(g, seed=np.random),
        "avg_neighbor_degree": nx.average_neighbor_degree,
        "closeness_centrality": nx.closeness_centrality,
        "number_of_nodes": lambda g: [nx.number_of_nodes(g)] * nx.number_of_nodes(g)
    }

    def make_data(self, G):
        """Create a PyG data object from a graph object."""

        # Compute and add features to the nodes in the graph.
        for feature in self.feature_functions:
            feature_val = self.feature_functions[feature](G)
            for node in G.nodes():
                for feature in self.feature_functions:
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

            #data[-1].y = labels
        #print(data)
        #torch_G.y = torch.cat(true_labels) # all solutions
        #torch_G.y = true_labels[-1] # one solution

        #data[:] = [x / len(true_labels) for x in data]
        torch_G.y = torch.div(data,count)#/len(true_labels) # scaled all solutions into one

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
    selected_graph_sizes = {#3: -1,
                            #4: -1,
                            #5: -1,
                            #6: -1,
                            #7: -1,
                            8: -1,
                            9:  10000,
                            #10: 1000,
                            #11: 1000,
                            #12: 1000,
                            #13: 1000,
                            #14: 1000,
                            #15: 1000,
                            #20: 2000,
                            #30: 10000,
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
