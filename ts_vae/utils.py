from rdkit import Chem
import pymol
import tempfile, os
import numpy as np

import torch


from math import sqrt, pi


from torch_sparse import SparseTensor

# what other TS properties do I need? syndags has good code for properties.

# ts_gen have pymol render but one of their imports tempfile doesn't exist anymore?



def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

### distance geometry funcs

def cartesian_to_new(pos, edge_index, num_nodes):
    # https://github.com/divelab/DIG/blob/dig/dig/threedgraph/utils/geometric_computing.py

    j, i = edge_index

    # dist sqrt(dx^2 + dy^2 + dz^2)
    dists = (pos[i] - pos[j]).pow(2).sum(dim = -1).sqrt()

    # TODO: pass in device of model
    value = torch.arange(j.size(0), device = 'cpu')
    # 
    adj_t = SparseTensor(row = i, col = j, value = value, sparse_sizes = (num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    # num_triplets ... TODO

    # node indices
    # ...

    # edge indices
    # ...

    # angles
    # ...

    # maybe torsion angles
    # ...

    return


