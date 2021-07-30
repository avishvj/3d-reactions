import torch
import torch.nn as nn
from torch_scatter import scatter_sum

# class global constants, TODO: move these into sep file
NUM_EDGE_ATTR = 3
MAX_N = 21

class GNN(nn.Module):
    """PyTorch version of MIT's ts_gen GNN model https://github.com/PattanaikL/ts_gen."""
    
    def __init__(self, in_node_nf, in_edge_nf, h_nf = 100, n_layers = 2, gnn_depth = 3):
        super(GNN, self).__init__()
        self.h_nf = h_nf
        self.n_layers = n_layers
        self.gnn_depth = gnn_depth

        # init layers
        self.node_mlp = MLP(in_node_nf, h_nf, n_layers)
        self.edge_mlp = MLP(in_edge_nf, h_nf, n_layers)
        
        # edge layers 
        self.pf_layer = PairFeaturesLayer(h_nf)
        self.de_mlp = MLP(h_nf, h_nf, n_layers) 

        # node layers
        self.dv_mlp1 = MLP(h_nf, h_nf, n_layers)
        self.dv_mlp2 = MLP(h_nf, h_nf, n_layers)

    def forward(self, node_feats, edge_attr, batch_size, mask_V, mask_E):
        
        # init hidden states of nodes and edges
        node_feats = mask_V * self.node_mlp(node_feats)
        edge_attr = mask_E * self.edge_mlp(edge_attr)

        # iteratively update edges (pair features, MLP, set final), nodes (MLP, reduce, MLP, set final)
        for _ in range(self.gnn_depth):
            edge_attr = self.edge_model(node_feats, edge_attr, batch_size, mask_E)
            # node_feats = self.node_model(node_feats, edge_attr, batch_size, mask_V)

        return node_feats, edge_attr

    def edge_model(self, node_feats, edge_attr, batch_size, mask_E):
        f = self.pf_layer(node_feats, edge_attr, batch_size)
        dE = self.de_mlp(f)
        return edge_attr + mask_E * dE

    def node_model(self, node_feats, edge_attr, batch_size, mask_V):
        dV = self.dv_mlp1(edge_attr)
        # reshape out from (batch_size * MAX_N**2, h_nf)
        dV = dV.view(batch_size, MAX_N, MAX_N, self.h_nf)
        dV = torch.sum(dV, 2).view(batch_size * MAX_N, self.h_nf)
        dV = self.dv_mlp2(dV)
        return node_feats + mask_V * dV


class PairFeaturesLayer(nn.Module):

    def __init__(self, h_nf, act = nn.ReLU()):
        super(PairFeaturesLayer, self).__init__()
        self.h_nf = h_nf
        self.edge_ij_layer = nn.Linear(h_nf, h_nf, bias = True)
        self.node_i_layer = nn.Linear(h_nf, h_nf, bias = False) 
        self.node_j_layer = nn.Linear(h_nf, h_nf, bias = False) 
        self.act = act

    def forward(self, node_feats, edge_attr, batch_size):
        # edge_attr input dim: N^2xEA
        f_ij = self.edge_ij_layer(edge_attr)
        f_i = self.node_i_layer(node_feats)
        f_j = self.node_j_layer(node_feats)

        # unsqueeze for final addition, then squeeze back again
        f_ij = f_ij.view(batch_size, MAX_N, MAX_N, self.h_nf)
        f_i = torch.unsqueeze(f_i.view(batch_size, MAX_N, self.h_nf), 1)
        f_j = torch.unsqueeze(f_j.view(batch_size, MAX_N, self.h_nf), 2)
        s = self.act(f_ij + f_i + f_j).view(batch_size * MAX_N**2, self.h_nf) 
        return s


class MLP(nn.Module):
    # add norm layers?

    def __init__(self, in_nf, out_nf, n_layers, act = nn.ReLU()):
        super(MLP, self).__init__()
        h_nf = in_nf

        self.layers = nn.ModuleList()
        for layer in range(n_layers):
            if layer == 0:
                self.layers.append(nn.Linear(in_nf, h_nf))
            else:
                self.layers.append(nn.Linear(h_nf, h_nf))
            self.layers.append(act)
        self.layers.append(nn.Linear(h_nf, out_nf))
        self.num_layers = len(self.layers)

    def forward(self, feats):
        for i in range(self.num_layers):
            feats = self.layers[i](feats)
        return feats
        

