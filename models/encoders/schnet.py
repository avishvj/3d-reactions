import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Embedding

from torch_geometric.nn import radius_graph
from torch_scatter import scatter

from math import pi as PI

from ts_ae.tsae import RPEncoder_Parent


# pytorch version of schnet, lifted from pytorch geometric and dig
# https://github.com/divelab/DIG/blob/dig/dig/threedgraph/method/schnet/schnet.py
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/schnet.html


# define edge and node model as whichever one finishes its model first
# will have to make this an AE


class SchNet_RPEncoder(RPEncoder_Parent):

    def __init__(self, in_node_nf, in_edge_nf, h_nf, out_nf, emb_nf, device):
        super(SchNet_RPEncoder, self).__init__(in_node_nf, in_edge_nf, h_nf, out_nf, emb_nf, device)
    
    def encode(self, batch):
        # get batch node_feats, edge_index, edge_attr, coords
        # create node and edge embs
        # mean pool to get graph emb
        return graph_emb


class SchNetEncoder(nn.Module):

    def __init__(self, h_nf, n_layers, n_filters, n_gaussians, cutoff_val):
        super(SchNetEncoder, self).__init__()
        self.h_nf = h_nf
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.n_gaussians = n_gaussians

        # init
        self.init_node_emb = Embedding(100, h_nf) # NOTE: is this 100 because upper bound of possible chemical elems?
        self.init_dist_emb = GaussianSmearing(0., cutoff_val, n_gaussians) # NOTE: make 0. const

        # edge update
        self.node_lin = Linear(h_nf, n_filters, bias = False) # TODO: change name
        self.cutoff_net = CosineCutoff(cutoff_val)
        self.edge_mlp = Sequential(Linear(n_gaussians, n_filters), ShiftedSoftplus(), Linear(n_filters, n_filters))

        # node update
        self.node_mlp = Sequential(Linear(n_filters, h_nf), ShiftedSoftplus(), Linear(h_nf, h_nf))

        # graph update
        self.graph_mlp = Sequential(Linear(h_nf, h_nf // 2), ShiftedSoftplus(), Linear(h_nf // 2, 1))
    
    def forward(self, atomic_ns, coords, batch_node_vec):
        # == interaction block pass
        
        # create node feats from atom charges
        node_feats = self.init_node_emb(atomic_ns)
        
        # create edge attrs from coordinates
        edge_index = radius_graph(coords, self.cutoff, batch_node_vec)
        node_is, node_js = edge_index
        edge_weights = (coords[node_is] - coords[node_js]).norm(dim = -1)
        edge_attr = self.init_dist_emb(edge_weights) # edge_attr = interatomic dist embs

        for _ in range(self.n_layers):
            edge_attr = self.edge_model(node_feats, edge_weights, edge_attr, node_is)
            node_feats = self.node_model(node_feats, edge_attr, node_js)
    
        return self.graph_model(node_feats, batch_node_vec)

    def node_model(self, node_feats, edge_attr, node_js):
        # schnet diagram: TODO
        node_out = scatter(edge_attr, node_js, dim = 0)
        node_out = self.node_mlp(node_out)
        return node_feats + node_out
    
    def edge_model(self, node_feats, edge_weights, edge_attr, node_is):
        # schnet diagram: first node layer + cfconv post init
        # edge_weights are interatomic dists, edge_attr are interatomic dist embs
        node_feats = self.node_lin(node_feats)
        cutoff = self.cutoff_net(edge_weights)
        W = self.edge_mlp(edge_attr) * cutoff.view(-1, 1)
        return node_feats[node_is] * W
    
    def graph_model(self, node_feats, node_batch_vec):
        # schnet diagram: blue blocks
        node_feats = self.graph_mlp(node_feats)
        graph_emb = scatter(node_feats, node_batch_vec, dim = 0) # TODO: double check scatter
        return graph_emb


class CosineCutoff(nn.Module):
    # following 3DMP, we use the cosine cutoff
    # also hard cutoff and mollifier cut off

    def __init__(self, cutoff = 0.5):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor[cutoff])
    
    def forward(self, interatomic_dists):
        # compute cutoffs and remove those beyond cutoff radius
        cutoffs = 0.5 * (torch.cos(interatomic_dists * PI / self.cutoff) + 1.0)
        cutoffs *= (interatomic_dists < self.cutoff).float()
        return cutoffs


class GaussianSmearing(nn.Module):
    # expand interatomic distances to get interatomic distance embeddings
    # this is one method but there are others too

    def __init__(self, start, stop, n_gaussians):
        super(GaussianSmearing, self).__init__()
        # offset of gaussian funcs, NOTE: could set trainable
        offset = torch.linspace(start, stop, n_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
    
    def forward(self, interatomic_dists):
        interatomic_dists = interatomic_dists.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(interatomic_dists, 2))


class ShiftedSoftplus(nn.Module):
    # the act function they use

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.)).item()
    
    def forward(self, x):
        return F.softplus(x) - self.shift