import networkx as nx
import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
import torch
import torch_geometric.utils as tg_utils
from my_graphs_dataset import GraphDataset


def find_MIDS(G):
    """
    Find all Minimum Independent Dominating Sets in a graph G.

    Uses Bron-Kerbosch algorithm to find all maximal cliques in the complement of G.
      Minimum Independent Dominating Set == Smallest Maximal Independent Set
      Maximal Independent Set == Maximal Clique of the complement
      ===
      Minimum Independent Dominating Set == Smallest Maximal Clique of the complement
    """
    Gc = nx.complement(G)
    min_size = len(Gc)
    min_cliques = []
    for nodes in nx.find_cliques(Gc):
        size = len(nodes)
        if size < min_size:
            min_size = size
            min_cliques = [nodes]
        elif size == min_size:
            min_cliques.append(nodes)
    return min_cliques


def check_MIDS(A, candidate, target_value):
    """
    Check if the candidate set is a MIDS.

    Args:
        - A: adjacency matrix
        - candidate: node labels that are candidate for MIDS (data.y)
        - target_value: known size of the MIDS
    """
    # TODO: This function needs to be adjusted.
    #   - Instead of adjacency matrix, we may pass the edgelist and convert it here

    n = len(candidate)

    # Candidate set is not minimal
    if sum(candidate) > target_value:
        return False

    # Candidate set is not dominating.
    if not all((A + np.eye(n)) @ candidate >= 1):
        return False

    # Candidate set is not independent.
    for i in range(n):
        for j in range(i + 1, n):
            if candidate[i] == 1 and candidate[j] == 1 and A[i, j] == 1:
                return False

    if sum(candidate) < target_value:
        print("Somehow we found an even smaller MIDS.")

    return True

def check_MIDS_batch(data, pred):
    """
    Args:
        data_batch (DataBatch): A batch of graphs from PyTorch Geometric.
        candidates (torch.Tensor): Candidate sets for MIDS, a 0-1 tensor indicating nodes in MIDS for each graph.

    Returns:
        torch.Tensor: A boolean tensor indicating if each candidate set satisfies the MIDS conditions.
    """
    ## Step 0: Initialize variables
    batch_size = data.batch.max().item() + 1
    node_batch = data.batch
    row, _ = data.edge_index
    edge_batch = data.batch[row]
    target = data.y if data.y.dim() == 1 else data.y[:, 0]

    device = data.y.device
    dtype = torch.int

    pred = pred.to(torch.int)

    ## Step 1: Check candidate set size condition
    # Sum up all target values across the batch.
    # Floating point errors are possible so we round to the nearest integer (diffrence is in the order <1e-3).
    target_values = torch.zeros(batch_size, dtype=data.y.dtype, device=device).scatter_add_(0, node_batch, target)
    target_values = torch.round(target_values).to(torch.int)

    mids_sizes = torch.zeros(batch_size, dtype=dtype, device=device).scatter_add_(0, node_batch, pred)
    size_condition = mids_sizes <= target_values

    ## Step 2: Check if candidate set is dominating
    # Add self-loops so that each node counts itself as a neighbor.
    edge_index_with_loops = tg_utils.add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]

    # Propagate the candidate values to each node’s neighbors.
    dom_neighbors = torch.zeros(data.num_nodes, dtype=dtype, device=device)
    dom_neighbors.index_add_(0, edge_index_with_loops[0], pred[edge_index_with_loops[1]])
    pred_dominance = (dom_neighbors >= 1).to(torch.int)

    # Aggregate per graph to ensure all nodes are dominated
    domination_per_graph = torch.zeros(batch_size, dtype=dtype, device=device).scatter_add_(0, node_batch, pred_dominance)
    domination_condition = domination_per_graph == torch.bincount(node_batch, minlength=batch_size)

    ## Step 3: Check independence condition
    # Mask edges where both nodes are in the candidate set
    edge_candidates = torch.logical_not(torch.logical_and(pred[data.edge_index[0]], pred[data.edge_index[1]])).to(torch.int)
    independence_per_graph = torch.zeros(batch_size, dtype=dtype, device=device).scatter_add_(0, edge_batch, edge_candidates)
    independence_condition = independence_per_graph == torch.bincount(edge_batch, minlength=batch_size)

    # Combine all conditions
    mids_conditions = size_condition & domination_condition & independence_condition

    return mids_conditions

def disjunction_value(G):
    """Calculate the disjunction value of each node in the graph G."""
    A = nx.to_numpy_array(G)
    nodes = list(G.nodes)
    n = A.shape[0]
    s = np.linalg.lstsq(A + np.eye(n), np.ones(n), rcond=None)[0]
    s = np.round(s, 2)
    return {nodes[i]: s[i] for i in range(n)}


def test_find_MIDS():
    """Test the MIDS finding algorithm."""
    def to_vector(nodes, num_nodes):
        # Encode found cliques as support vectors.
        mids = np.zeros(num_nodes)
        mids[nodes] = 1
        return mids

    loader = GraphDataset()

    for G in loader.graphs():
        A = nx.to_numpy_array(G)
        # nx.draw(G, with_labels=True)
        # plt.show()
        possible_MIDS = find_MIDS(G)

        for MIDS in possible_MIDS:
            np_MIDS = to_vector(MIDS, G.number_of_nodes())
            if not check_MIDS(A, np_MIDS, len(MIDS)):
                print(MIDS, np_MIDS)
                pos = nx.spring_layout(G)
                plt.figure()
                nx.draw(G, with_labels=True, node_color=np_MIDS, cmap=colormaps["bwr"], pos=pos)
                plt.figure()
                Gc = nx.complement(G)
                nx.draw(Gc, with_labels=True, node_color=np_MIDS, cmap=colormaps["bwr"], pos=pos)
                plt.show()


if __name__ == "__main__":
    test_find_MIDS()
