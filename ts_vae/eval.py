import torch
import matplotlib.pyplot as plt

# TODO: edge diff in reaction core. otherwise, will get really high edge diff

def adj_error(adj_pred, adj_gen):
    # probabilistic adj, sum adj_errors, 

    num_nodes = adj_gen.size(0)
    adj_pred = (adj_pred > 0.5).type(torch.float32)
    adj_errors = torch.abs(adj_pred - adj_gen)
    wrong_edges = torch.sum(adj_errors)
    adj_error = wrong_edges / (num_nodes ** 2 - num_nodes)
    return wrong_edges.item(), adj_error.item()

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