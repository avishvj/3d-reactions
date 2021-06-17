

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

from .layers import NELayer, NECLayer, GCL_PYG

class EGNN(nn.Module):
    def __init__(self, n_layers, feats_dim, pos_dim = 3, edge_attr_dim = 0, latent_dim = 2,
                 embedding_nums = [], embedding_dims = [], edge_embedding_nums = [], edge_embedding_dims = [],
                 update_coords = True, update_feats = True, norm_feats = True, norm_coords = False, norm_coords_scale_init = 1e-2,
                 aggr = "add"):
        
        super().__init__()

        self.n_layers = n_layers

        # emb
        self.embedding_nums = embedding_nums
        self.embedding_dims = embedding_dims
        self.emb_layers = nn.ModuleList()
        self.edge_embedding_nums = edge_embedding_nums
        self.edge_embedding_dims = edge_embedding_dims
        self.edge_emb_layers = nn.ModuleList()

        # instantiate node and edge embedding layers
        for i in range(len(self.embedding_dims)):
            self.emb_layers.append(nn.Embedding(num_embeddings = embedding_nums[i],
                                                embedding_dim = embedding_dims[i]))
            feats_dim += embedding_dims[i] - 1
        for i in range(len(self.edge_embedding_dims)):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings = edge_embedding_nums[i],
                                                     embedding_dim = edge_embedding_dims[i]))
            edge_attr_dim += edge_embedding_dims[i] - 1
        
        # other params
        self.mpnn_layers = nn.ModuleList()
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.edge_attr_dim = edge_attr_dim
        self.latent_dim = latent_dim
        self.norm_feats = norm_feats
        self.norm_coords = norm_coords
        self.norm_coords_scale_init = norm_coords_scale_init
        self.update_feats = update_feats
        self.update_coords = update_coords
        
        # instantiate layers
        for i in range(n_layers):
            egnn_layer = GCL_PYG(feats_dim = feats_dim, pos_dim = pos_dim, edge_attr_dim = edge_attr_dim,
                                 latent_dim = latent_dim, norm_feats = norm_feats, norm_coords = norm_coords,
                                 norm_coords_scale_init = norm_coords_scale_init, update_feats = update_feats, update_coords = update_coords)
            self.mpnn_layers.append(egnn_layer)
        
    def forward(self, x, edge_index, batch, edge_attr):
        # TODO: recalc edge features every recalc_edge

        # nodes
        x = embedding_token(x, self.embedding_dims, self.emb_layers)

        # regulates whether to embedding edges each layer
        edges_need_embedding = True
        for i, layer in enumerate(self.mpnn_layers):

            # edges
            if edges_need_embedding:
                edge_attr = embedding_token(edge_attr, self.edge_embedding_dims, self.edge_emb_layers)
                edge_need_embedding = False

            # pass layers
            x = layer(x, edge_index, edge_attr, batch = batch, size = bsize)

        return x

    def __repr__(self):
        return "EGNN Network of: {0} layers".format(len(self.mpnn_layers))

def embedding_token(x, dims, layers):
    stop_concat = -len(dims)
    to_embedding = x[:, stop_concat: ].long()
    for i, emb_layer in enumerate(layers):

        # portion corresponding to 'to_embedding' gets dropped
        x = torch.cat([x[:, : stop_concat], emb_layer(to_embedding[:, i])], dim = -1)
        stop_concat = x.shape[-1]
    return x


class EGNN_Simple(nn.Module):

    def __init__(self, in_nf, hidden, out, in_edge, device = 'cpu', act_fn = nn.ReLU(), 
                 n_layers = 1, norm_coords = False):
        
        super(EGNN_Simple, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.emb_in = nn.Linear(in_nf, hidden)
        self.emb_out = nn.Linear(hidden, out)

        for i in range(0, n_layers):
            self.add_module("GCL_Simple_%d" % i, NECLayer(self.hidden, self.hidden, self.hidden, edges_in_dim = in_edge,
                            act_fn = act_fn, norm_coords = norm_coords))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        h = self.emb_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self.modules["GCL_Simple_%d" % i](h, edges, x, edge_attr = edge_attr)
        h = self.emb_out(h)
        return h, x


class EGNN_AE(nn.Module):
    # simple one first

    def __init__(self, h_nf, emb_nf = 4, num_node_fs = 11, n_layers = 1, act_fn = nn.ReLU(), device = 'cpu'):
        super(EGNN_AE, self).__init__()

        self.num_node_fs = num_node_fs
        self.h_nf = h_nf
        self.emb_nf = emb_nf
        self.n_layers = n_layers
        self.device = device
        
        # encoder
        ne_layer = NELayer(in_nf = num_node_fs, out_nf = h_nf, h_nf = h_nf, edges_nf = 4, act_fn = act_fn)
        self.add_module("NE", ne_layer)
        self.fc_emb = nn.Linear(h_nf, emb_nf)

        # decoder linear layer W and b
        self.W = nn.Parameter(-0.1 * torch.ones(1)).to(device)
        self.b = nn.Parameter(torch.ones(1)).to(device)
        
        self.to(device)
    
    def forward(self, node_feats, edge_index, edge_attr = None):        
        # encode features then decode to adj
        # could just encode features to coords then decode to adj

        node_feats, edge_feats = self.encode(node_feats, edge_index, edge_attr)
        return self.fc_emb(node_feats)

        # adj_pred = self.decode(x) # TODO: sort out!!!
        # return adj_pred

    def encode(self, node_feats, edge_index, edge_attr):
        # one (node + edge) layer
        node_feats, edge_feats = self._modules["NE"](node_feats, edge_index, edge_attr)
        return node_feats, edge_feats
    
    def decode(self, x):
        return self.decode_from_x(x, W = self.W, b = self.b)

    def decode_from_x(self, x, W = 10, b = -1, remove_diag = True):
        # W, b: weights and biases in case no linear layer
        x_a = x.unsqueeze(0)
        x_b = torch.transpose(x_a, 0, 1) # (_, first dim to t(), second_dim to t())
        X = (x_a - x_b) ** 2
        
        num_nodes = x.size(0)
        X = X.view(num_nodes ** 2, -1) 
        X = torch.sigmoid(W * torch.sum(X, dim = 1) + b) # linear layer
        
        adj_pred = X.view(num_nodes, num_nodes)
        if remove_diag: # TODO: the pyg method adds self-loops, what do I want?
            adj_pred = adj_pred * (1 - torch.eye(num_nodes).to(self.device))
        
        return adj_pred


class EGNN_NEC(nn.Module):

    def __init__(self, in_nf, out_nf, h_nf, in_ef, device = 'cpu', act_fn = nn.ReLU(), 
                 n_layers = 1, norm_coords = False):
        # nf: node features; ef: edge features
        
        super(EGNN_NEC, self).__init__()

        self.h_nf = h_nf
        self.device = device
        self.n_layers = n_layers
        self.emb_in = nn.Linear(in_nf, h_nf)
        self.emb_out = nn.Linear(h_nf, out_nf)

        # one NEC layer for now
        self.add_module("NEC", NECLayer(in_nf = h_nf, out_nf = h_nf, hidden_nf = h_nf,
                        edges_in_dim = in_ef, act_fn = act_fn, norm_coords = norm_coords))
        
        self.to(device)

    def forward(self, node_feats, edge_index, edge_attr, coords):
        # emb_in(in, hidden) -> NEC(hidden, hidden) -> emb_out(hidden, out)
        node_feats = self.emb_in(node_feats)
        # one NEC layer
        node_feats, edge_feats, coords = self._modules["NEC"](node_feats, 
                                                        edge_index, edge_attr, coords)
        node_feats = self.emb_out(node_feats)
        return node_feats, coords
    
        


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
        # TODO: what is h???
        
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
        
        # .view() reshapes tensor into desired dim
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




    