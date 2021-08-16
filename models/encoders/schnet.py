from math import pi as PI
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Embedding
from torch_geometric.nn import radius_graph
from torch_scatter import scatter

class SchNetEncoder(nn.Module):
    """PyTorch version of SchNet, mostly lifted from PyTorch Geometric and DIG implementations.
    Sources:
        - SchNet paper: https://arxiv.org/abs/1706.08566
        - PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/schnet.html
        - DIG: https://github.com/divelab/DIG/blob/dig/dig/threedgraph/method/schnet/schnet.py
    """

    def __init__(self, h_nf, n_filters, n_gaussians, cutoff_val, n_layers = 1):
        super(SchNetEncoder, self).__init__()

        self.init = SchNetInit(h_nf, cutoff_val, n_gaussians)

        self.n_layers = n_layers
        for l in range(self.n_layers):
            self.add_module(f"SchNet_{l}", SchNetUpdate(h_nf, n_filters, n_gaussians, cutoff_val))

        self.post = SchNetPost(h_nf)
    
    def forward(self, batch):
        """One pass of the SchNet interaction block."""
        atomic_ns, batch_node_vec = batch['atomic_ns'], batch['batch_node_vec']
        edge_index = batch['edge_index']
        coords = batch['coords']
        node_embs, edge_embs, edge_weights = self.init(atomic_ns, edge_index, coords, batch_node_vec)
        for l in range(self.n_layers):
            node_embs, edge_embs = self._modules[f"SchNet_{l}"](node_embs, edge_index, edge_embs, edge_weights)
        node_embs, graph_emb = self.post(node_embs, batch_node_vec)
        return node_embs, graph_emb, None # None for coords

### Main classes used for SchNet processing

class SchNetInit(nn.Module):
    """Initialise node and edge (dist) embeddings to pass to looped SchNet layers."""
    POSS_ELEMS = 100
    ORIGIN = 0.
    
    def __init__(self, h_nf, cutoff_val, n_gaussians):
        self.node_emb = Embedding(self.POSS_ELEMS, h_nf) # NOTE: is this 100 because upper bound of possible chemical elems?
        self.dist_emb = GaussianSmearing(self.ORIGIN, cutoff_val, n_gaussians) # NOTE: make 0. const
    
    def forward(self, atomic_ns, edge_index, coords, batch_node_vec):
        
        # create initial node embs from atom charges
        node_embs = self.node_emb(atomic_ns)
        
        # create initial edge embs from coordinates
        edge_index = radius_graph(coords, self.cutoff, batch_node_vec)
        node_is, node_js = edge_index
        edge_weights = (coords[node_is] - coords[node_js]).norm(dim = -1)
        edge_embs = self.dist_emb(edge_weights) # edge_embs = interatomic dist embs

        return node_embs, edge_embs, edge_weights

class SchNetUpdate(nn.Module):
    """SchNet update block. Subblock of overall interaction block that can be looped."""

    def __init__(self, h_nf, n_filters, n_gaussians, cutoff_val):

        # edge update
        self.node_lin = Linear(h_nf, n_filters, bias = False) 
        self.cutoff_net = CosineCutoff(cutoff_val)
        self.edge_mlp = Sequential(Linear(n_gaussians, n_filters), ShiftedSoftplus(), Linear(n_filters, n_filters))

        # node update
        self.node_mlp = Sequential(Linear(n_filters, h_nf), ShiftedSoftplus(), Linear(h_nf, h_nf))
    
    def forward(self, node_embs, edge_index, edge_embs, edge_weights):
        """Updates node and edge embeddings. edge_weights are IA dists, edge_embs are IA dist embs.
        NOTE: Why node_is and node_js?
        """
        node_is, node_js = edge_index
        edge_embs = self.edge_update(node_embs, node_is, edge_embs, edge_weights)
        node_embs = self.node_update(node_embs, node_js, edge_embs)
        return node_embs, edge_embs
    
    def node_update(self, node_embs, node_js, edge_embs):
        """SchNet diagram: TODO"""
        node_out = scatter(edge_embs, node_js, dim = 0)
        node_out = self.node_mlp(node_out)
        return node_embs + node_out
    
    def edge_update(self, node_embs, node_is, edge_embs, edge_weights):
        """SchNet diagram: first node layer + cfconv post init."""
        node_embs = self.node_lin(node_embs)
        cutoff = self.cutoff_net(edge_weights)
        W = self.edge_mlp(edge_embs) * cutoff.view(-1, 1)
        return node_embs[node_is] * W

class SchNetPost(nn.Module):
    """SchNet diagram: blue blocks. Create final node embeddings and graph embedding."""

    def __init__(self, h_nf):
        self.node_emb_out = Sequential(Linear(h_nf, h_nf // 2), ShiftedSoftplus(), Linear(h_nf // 2, 1))
    
    def forward(self, node_feats, node_batch_vec):
        node_embs = self.node_emb_out(node_feats)
        graph_emb = scatter(node_embs, node_batch_vec, dim = 0)
        return node_embs, graph_emb

### Subclasses used for SchNet processing

class CosineCutoff(nn.Module):
    """Following SphereNet, we also use the cosine cutoff. Alternatives: hard, mollifer cutoffs."""
    
    def __init__(self, cutoff = 0.5):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor[cutoff])
    
    def forward(self, interatomic_dists):
        # compute cutoffs and remove those beyond cutoff radius
        cutoffs = 0.5 * (torch.cos(interatomic_dists * PI / self.cutoff) + 1.0)
        cutoffs *= (interatomic_dists < self.cutoff).float()
        return cutoffs

class GaussianSmearing(nn.Module):
    """Expand interatomic distances to get interatomic distance embs. Alternative methods exist."""
    
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