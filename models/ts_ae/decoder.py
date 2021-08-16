###
import torch
import torch.nn as nn

class MolDecoder(nn.Module):

    def __init__(self, in_node_nf, in_edge_nf, emb_nf, device = 'cpu'):
        super(MolDecoder, self).__init__()
        
        # standard params
        self.in_node_nf = in_node_nf
        self.in_edge_nf = in_edge_nf
        self.emb_nf = emb_nf

        # mlps: node, edge
        self.node_dec_mlp = nn.Linear(emb_nf, in_node_nf)
        self.edge_dec_mlp = nn.Linear(emb_nf, in_edge_nf)
        # decoder adj [found these worked well]
        self.W = nn.Parameter(0.5 * torch.ones(1)).to(device)
        self.b = nn.Parameter(0.8 * torch.ones(1)).to(device)

        self.to(device)
    
    def forward(self, node_emb, edge_emb, graph_emb, coords):
        return self.decode(node_emb, edge_emb)

    def decode(self, node_emb, edge_emb):
        # decode to node_fs, edge_fs, adj
        recon_node_fs = self.node_dec_mlp(node_emb)
        recon_edge_fs = self.edge_dec_mlp(edge_emb)
        adj_pred = self.decode_to_adj(node_emb)
        return recon_node_fs, recon_edge_fs, adj_pred
    
    def decode_to_adj(self, x, remove_self_loops = True):
        # x dim: [num_nodes, 2], use num_nodes as adj_matrix dim
        # returns probabilistic adj matrix

        # create params from x
        num_nodes = x.size(0)
        x_a = x.unsqueeze(0) # dim: [1, num_nodes, 2]
        x_b = torch.transpose(x_a, 0, 1) # dim: [num_nodes, 1, 2], t.t([_, dim to t, dim to t])

        # generate diffs between node embs as adj matrix
        X = (x_a - x_b) ** 2 # dim: [num_nodes, num_nodes, 2]
        X = X.view(num_nodes ** 2, -1) # dim: [num_nodes^2, 2] to apply sum
        X = torch.sigmoid(self.W * torch.sum(X, dim = 1) + self.b) # sigmoid here since can get negative values with W, b
        # X = torch.tanh(torch.sum(X, dim = 1)) # no linear since can get negative values, gives better output but need diff way of training
        adj_pred = X.view(num_nodes, num_nodes) # dim: [num_nodes, num_nodes] 

        if remove_self_loops:
            adj_pred = adj_pred * (1 - torch.eye(num_nodes))

        return adj_pred

class TSDecoder(nn.Module):

    def __init__(self, device = 'cpu'):
        super(TSDecoder, self).__init__()
        
        # decoder adj [found these worked well]
        self.W = nn.Parameter(0.5 * torch.ones(1)).to(device)
        self.b = nn.Parameter(0.8 * torch.ones(1)).to(device)
        
        self.to(device)
    
    def forward(self, embs):
        node_embs, graph_embs = embs
        return self.decode_to_dist(node_embs)

    def decode_to_dist(self, x, remove_self_loops = True):
            # TODO: decode to coordinates directly!

            # x dim: [num_nodes, 2], use num_nodes as adj_matrix dim
            # returns probabilistic adj matrix

            # create params from x
            num_nodes = x.size(0)
            x_a = x.unsqueeze(0) # dim: [1, num_nodes, 2]
            x_b = torch.transpose(x_a, 0, 1) # dim: [num_nodes, 1, 2], t.t([_, dim to t, dim to t])

            # generate diffs between node embs as adj matrix
            X = (x_a - x_b) ** 2 # dim: [num_nodes, num_nodes, 2]
            X = X.view(num_nodes ** 2, -1) # dim: [num_nodes^2, 2] to apply sum
            X = torch.sigmoid(self.W * torch.sum(X, dim = 1) + self.b) # sigmoid here since can get negative values with W, b
            # X = torch.tanh(torch.sum(X, dim = 1)) # no linear since can get negative values, gives better output but need diff way of training
            adj_pred = X.view(num_nodes, num_nodes) # dim: [num_nodes, num_nodes] 

            if remove_self_loops:
                adj_pred = adj_pred * (1 - torch.eye(num_nodes))

            return adj_pred
