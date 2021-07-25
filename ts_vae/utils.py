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

### remove processed files

import os
import glob

def remove_files():
    files = glob.glob(r'data/processed/*')
    for f in files:
        os.remove(f)
    print("Files removed.")    

### aggregation funcs

def unsorted_segment_sum(edge_attr, row, num_segments):
    result_shape = (num_segments, edge_attr.size(1))
    result = edge_attr.new_full(result_shape, 0) # init empty result tensor
    row = row.unsqueeze(-1).expand(-1, edge_attr.size(1))
    result.scatter_add_(0, row, edge_attr) # adds all values from tensor other int self at indices
    return result

### distance geometry funcs

import torch
from torch_sparse import SparseTensor

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


import numpy as np
def interatomic_distance(atom_1, atom_2):
    return np.linalg.norm(atom_1.position - atom_2.position)

def interatomic_distance_matrix_initialise(mol):
    num_atoms = mol.GetNumAtoms()
    matrix = np.zeros(shape=(num_atoms, num_atoms), dtype=np.float)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            # create triangular matrix
            matrix[i, j] = 1e3 # max_dist
            matrix[j, i] = 1e-3 # min dist


def sparse_to_dense_adj(num_nodes, edge_index):
    # TODO: pyg to_dense_adj() returns same but with added singleton dim 
    # i.e. pyg: [1, num_nodes, num_nodes]; this: [num_nodes, num_nodes]
    # think pyg method can also factor in edge_attr

    # edge_index is sparse_adj matrix (given in coo format for graph connectivity)
    sparse_adj = torch.cat([edge_index[0].unsqueeze(0), edge_index[1].unsqueeze(0)])
    # the values we put in at each tuple; that's why length of sparse_adj
    ones = torch.ones(sparse_adj.size(1)) 
    # FloatTensor() creates sparse coo tensor in torch format, then to_dense()
    dense_adj = torch.sparse.FloatTensor(sparse_adj, ones, torch.Size([num_nodes, num_nodes])).to_dense() # to_dense adds the zeroes needed
    return dense_adj

### eval funcs

def adj_error(adj_pred, adj_gen):
    # probabilistic adj, sum adj_errors, 

    num_nodes = adj_gen.size(0)
    adj_pred = (adj_pred > 0.5).type(torch.float32)
    adj_errors = torch.abs(adj_pred - adj_gen)
    wrong_edges = torch.sum(adj_errors)
    adj_error = wrong_edges / (num_nodes ** 2 - num_nodes)
    return wrong_edges.item(), adj_error.item()