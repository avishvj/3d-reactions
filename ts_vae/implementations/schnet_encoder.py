# pytorch version of schnet, lifted from pytorch geometric and dig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Embedding

from torch_geometric.nn import radius_graph
from torch_scatter import scatter


class SchNet(nn.Module):

    def __init__(self, h_nf, n_interactions, n_filters, n_gaussians, cutoff):
        super(SchNet, self).__init__()
        self.h_nf = h_nf
        self.n_interactions = n_interactions
        self.n_filters = n_filters
        self.n_gaussians = n_gaussians
        self.cutoff = cutoff
        # maybe add dipole, mean/std?, is interactions == n_layers?

        # init
        self.init_node_emb = Embedding(100, h_nf)
        self.init_dist_emb = GaussianSmearing(0., cutoff, n_gaussians)

        # node update
        self.node_mlp = Sequential(Linear(n_filters))

        # edge update
        self.edge_mlp = Sequential(Linear(n_gaussians, n_filters), ShiftedSoftplus(), Linear(n_filters, n_filters))
        self.edge_lin = Linear(h_nf, n_filters, bias = False)

        # graph update
        self.graph_lin1 = Linear(h_nf, h_nf // 2)
        self.act = ShiftedSoftplus()
        self.graph_lin2 = Linear(h_nf // 2, 1)
    
    def forward(self, atom_charges, coords, node_batch_vec):
        
        # create node feats from atom charges
        node_feats = self.init_node_emb(atom_charges)
        
        # create edge attrs from coordinates
        edge_index = radius_graph(coords, self.cutoff, node_batch_vec)
        row, col = edge_index
        edge_weights = (coords[row] - coords[col]).norm(dim = -1)
        edge_attr = self.init_dist_emb(edge_weights) # edge_attr = interatomic dist embs
    
    def node_model(self, node_feats, edge_attr, edge_index):
        return
    
    def edge_model(self, node_feats, edge_weights, edge_attr, edge_index):
        return
    
    def graph_model(self, node_feats, node_batch_vec):
        node_feats = self.graph_lin1(node_feats)
        node_feats = self.act(node_feats)
        node_feats = self.graph_lin2(node_feats)
        graph_emb = scatter(node_feats, node_batch_vec, dim = 0)
        return graph_emb


class GaussianSmearing(nn.Module):

    def __init__(self, start, stop, n_gaussians):
        super(GaussianSmearing, self).__init__()

        # offset of gaussian funcs, NOTE: could set trainable
        offset = torch.linspace(start, stop, n_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
    
    def forward(self, interatomic_dists):
        interatomic_dists = interatomic_dists.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(interatomic_dists, 2))


class ShiftedSoftplus(nn.Module):
    
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.)).item()
    
    def forward(self, x):
        return F.softplus(x) - self.shift