import networkx as nx
import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt

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
