# pytorch version of schnet, lifted from pytorch geometric and dig
# https://github.com/divelab/DIG/blob/dig/dig/threedgraph/method/schnet/schnet.py
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/schnet.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Embedding

from torch_geometric.nn import radius_graph
from torch_scatter import scatter

from math import pi as PI


class SchNet(nn.Module):

    def __init__(self, h_nf, n_layers, n_filters, n_gaussians, cutoff):
        super(SchNet, self).__init__()
        self.h_nf = h_nf
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.n_gaussians = n_gaussians
        self.cutoff = cutoff
        # maybe add dipole, mean/std?, is interactions == n_layers?

        # init
        self.init_node_emb = Embedding(100, h_nf) # NOTE: is this 100 because upper bound of possible chemical elems?
        self.init_dist_emb = GaussianSmearing(0., cutoff, n_gaussians) # NOTE: make 0. const

        # edge update
        self.node_lin = Linear(h_nf, n_filters, bias = False) # TODO: change name
        self.edge_mlp = Sequential(Linear(n_gaussians, n_filters), ShiftedSoftplus(), Linear(n_filters, n_filters))

        # node update
        self.node_mlp = Sequential(Linear(n_filters, h_nf), ShiftedSoftplus(), Linear(h_nf, h_nf))

        # graph update
        self.graph_mlp = Sequential(Linear(h_nf, h_nf // 2), ShiftedSoftplus(), Linear(h_nf // 2, 1))
    
    def forward(self, atom_charges, coords, node_batch_vec):
        
        # create node feats from atom charges
        node_feats = self.init_node_emb(atom_charges)
        
        # create edge attrs from coordinates
        edge_index = radius_graph(coords, self.cutoff, node_batch_vec)
        row, col = edge_index
        edge_weights = (coords[row] - coords[col]).norm(dim = -1)
        edge_attr = self.init_dist_emb(edge_weights) # edge_attr = interatomic dist embs

        for _ in range(self.n_layers):
            edge_attr = self.edge_model(node_feats, edge_weights, edge_attr, edge_index)
            node_feats = self.node_model(node_feats, edge_attr, edge_index)
    
        return self.graph_model(node_feats, node_batch_vec)
    
    def updated_node_model(self, node_feats, edge_weights, edge_attr, edge_index):
        # == interaction block
        node_is, node_js =  edge_index
        node_feats = self.node_lin(node_feats)
        edge_attr = self.updated_edge_model(node_feats, edge_weights, edge_attr, node_is)
        
        
        node_out = scatter(edge_attr, node_js, dim = 0)
        node_out = self.node_mlp(node_out)
        return node_feats + node_out
    
    def updated_edge_model(self, node_feats, edge_weights, edge_attr, node_is):
        # == cfconv post init i.e. rbf onwards
        C = 1 # TODO: make proper
        W = self.edge_mlp(edge_attr) * C.view(-1, 1)
        return node_feats[node_is] * W


    def node_model(self, node_feats, edge_attr, edge_index):
        _, node_js =  edge_index
        node_out = scatter(edge_attr, node_js, dim = 0)
        node_out = self.node_mlp(node_out)
        return node_feats + node_out
    
    def edge_model(self, node_feats, edge_weights, edge_attr, edge_index):
        # edge_weights are interatomic dist, edge_attr are interatomic dist embs
        node_is, _ = edge_index
        node_feats = self.node_lin()
        W = self.edge_mlp(edge_attr) # TODO: add C



        return
    
    def graph_model(self, node_feats, node_batch_vec):
        # schnet diagram: blue blocks
        node_feats = self.graph_mlp(node_feats)
        graph_emb = scatter(node_feats, node_batch_vec, dim = 0) # TODO: double check scatter
        return graph_emb


class GaussianSmearing(nn.Module):
    # used to get interatomic distance embeddings

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