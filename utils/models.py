import torch
import numpy as np

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

### aggregation funcs

def unsorted_segment_sum(edge_attr, row, num_segments):
    result_shape = (num_segments, edge_attr.size(1))
    result = edge_attr.new_full(result_shape, 0) # init empty result tensor
    row = row.unsqueeze(-1).expand(-1, edge_attr.size(1))
    result.scatter_add_(0, row, edge_attr) # adds all values from tensor other int self at indices
    return result

### distance geometry funcs

def X_to_dist(X):
    # create euclidean distance matrix from X
    # shapes: X = bx21x3, D = bx21x21
    Dsq = torch.square(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2))
    D = torch.sqrt(torch.sum(Dsq, dim=3) + 1E-3)
    return D

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

### eval funcs

def adj_error(adj_pred, adj_gen):
    # probabilistic adj, sum adj_errors, 

    num_nodes = adj_gen.size(0)
    adj_pred = (adj_pred > 0.5).type(torch.float32)
    adj_errors = torch.abs(adj_pred - adj_gen)
    wrong_edges = torch.sum(adj_errors)
    adj_error = wrong_edges / (num_nodes ** 2 - num_nodes)
    return wrong_edges.item(), adj_error.item()