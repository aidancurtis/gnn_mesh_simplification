import os
import sys

import numpy as np
import torch
import torch_geometric

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../data/tosca")

from gnn_mesh_simplification.datasets import TOSCA, Watertight

dataset = TOSCA(data_dir)
print(dataset)
