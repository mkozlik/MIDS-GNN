import os
print(os.getcwd())

import torch
import torch_geometric

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

#import codetiming
import numpy
import pandas
import plotly
import wandb

import my_graphs_dataset
