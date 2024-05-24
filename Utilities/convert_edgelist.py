import numpy as np
import networkx as nx
from pathlib import Path
from collections import OrderedDict


data_folder = Path.cwd() / "Data"
new_data = Path.cwd() / "NewData"

for i in range(3, 9):
    with open(data_folder / f"UniqueGraphs_{i}.npz", "rb") as stream:
        loaded = np.load(stream)
        gs = OrderedDict(loaded)

    save_path = new_data / f"{i}_nodes"
    save_path.mkdir(parents=True, exist_ok=True)
    for name, adj in gs.items():
        g = nx.from_numpy_array(adj)
        nx.write_edgelist(g, save_path / f"{name}.txt", data=False)
