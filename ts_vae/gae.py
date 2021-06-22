

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)
from sklearn.metrics import roc_auc_score, average_precision_score
from .utils import reset

EPS = 1e-15

class MolEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MolEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        z = self.conv(x, edge_index) 
        return z

class InnerProductDecoder(nn.Module):
    def forward(self, z, edge_index, sigmoid = True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim = 1)
        return torch.sigmoid(value) if sigmoid else value
    
    def forward_all(self, z, sigmoid = True):
        """ Decode latent variables into probabilistic adj matrix. """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

class GAE(nn.Module):
    """ Identical copy of the GAE given in PyTorch Geometric. """

    def __init__(self, encoder, decoder = None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder()
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder) 

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index = None):
        """ BCE loss between input adj matrix and reconstructed adj matrix.
        """
        pos_loss = - torch.log(self.decoder(z, pos_edge_index, sigmoid = True) + EPS).mean()

        # no self-loops in negative samples -> not sure if mini-batch is stochastic but this is definitely the reason for variation
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid = True) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        """ Take latent z and test_edge_indices, recreate adj matrix and compare to default.
        """

        # create 1s of pos edges and 0s of neg edges, then concat
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        # reconstruct pos edges and neg edges, then concat
        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


    