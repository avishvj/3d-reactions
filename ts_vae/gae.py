

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




### other way

class EdgeDecoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
    
    def forward(self, input_data, training = False):
        return self.decoder(input_data.edges, training = training)

class NodeDecoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
    
    def forward(self, input_data, training = False):
        return self.decoder(input_data.nodes, training = training)

# then could have CoordDecoder and GlobalDecoder, too

from .layers import GCL

class BasicAE(nn.Module):
    # TODO: replace x -> z when encoding

    def __init__(self, hidden_nf, embedding_nf, device = 'cpu', act_fn = nn.ReLU(), n_layers = 0):
        super(BasicAE, self).__init__()
        self.hidden_nf = hidden_nf
        self.embedding_nf = embedding_nf
        self.device = device
        self.n_layers = n_layers

        # encoder
        self.add_module("GCL_0", GCL(1, self.hidden_nf, self.hidden_nf, edges_in_nf = 1))
        for i in range(1, n_layers):
            self.add_module("GCL_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf = 1))
        self.fc_emb = nn.Linear(self.hidden_nf, self.embedding_nf)

        # decoder layer
        self.dec_layer = None
        
        self.to(self.device)

    def encode(self, nodes, edges, edge_attr):
        # note: assumes all layers are GCL layers
        
        # initial layer
        h, _ = self._modules["GCL_0"](nodes, edges, edge_attr)
        
        # for hidden layers if present
        for i in range(1, self.n_layers):
            h, _ = self._modules["GCL_%d" % i](h, edges, edge_attr = edge_attr) 
        
        return self.fc_emb(h)


    def decode(self, x):
        return self.decode_from_x(x, linear_layer = self.dec_layer)

    def decode_from_x(self, x, linear_layer = None, W = 10, b = -1, remove_diag= True):
        # W, b: weights and biases in case no linear layer
        
        num_nodes = x.size(0)
        x_a = x.unsqueeze(0)
        x_b = torch.transpose(x_a, 0, 1) # (_, first dim to t(), second_dim to t())
        X = (x_a - x_b) ** 2
        
        X = X.view(num_nodes ** 2, -1) # TODO: sigmoid?

        if linear_layer is not None:
            X = torch.sigmoid(linear_layer(X))
        else:
            X = torch.sigmoid(W * torch.sum(X, dim = 1) + b)
        
        adj_pred = X.view(num_nodes, num_nodes)
        if remove_diag:
            # eye returns 2D tensor with ones on diag and zeroes elsewhere
            # (1 - torch.eye(num_nodes)) gives [num_nodes, num_nodes] with all 1s except 0 on diag
            # * is hadamard product
            # i.e. this removes self loops (i.e. 1s on diag)
            # TODO: the pyg method adds self-loops, what do I want?
            adj_pred = adj_pred * (1 - torch.eye(num_nodes).to(self.device))
        
        return adj_pred

    def forward(self, nodes, edges, edge_attr = None):
        x = self.encode(nodes, edges, edge_attr)
        adj_pred = self.decode(x)
        return adj_pred, x




    