import torch
import torch.nn as nn
from torch_scatter import scatter_sum

class GNN(nn.Module):
    """PyTorch version of MIT's ts_gen GNN model https://github.com/PattanaikL/ts_gen. """
    NUM_EDGE_ATTR = 3

    def __init__(self, in_node_nf, in_edge_nf, h_nf = 100, n_layers = 2, num_iterations = 3):
        super(GNN, self).__init__()
        self.h_nf = h_nf
        self.n_layers = n_layers
        self.num_iterations = num_iterations

        # init layers
        self.node_mlp = MLP(in_node_nf, h_nf, n_layers)
        self.edge_mlp = MLP(in_edge_nf, h_nf, n_layers)
        
        # edge layers 
        self.pf_layer = PairFeaturesLayer(h_nf)
        self.de_mlp = MLP(h_nf, h_nf, n_layers) 

        # node layers
        self.dv_mlp1 = MLP(h_nf, h_nf, n_layers)
        self.dv_mlp2 = MLP(h_nf, h_nf, n_layers)

    def forward(self, node_feats, edge_attr):
        
        # reshape edge_attr from NxNxEA to N^2xEA for use in mlps, TODO: batch?
        N = len(node_feats)
        edge_attr = edge_attr.view(N**2, self.NUM_EDGE_ATTR)

        # init hidden states of nodes and edges
        node_feats = self.node_mlp(node_feats)
        edge_attr = self.edge_mlp(edge_attr)

        # iteratively update edges (pair features, MLP, set final), nodes (MLP, reduce, MLP, set final)
        for _ in range(self.num_iterations):
            edge_attr = self.edge_model(node_feats, edge_attr)
            node_feats = self.node_model(node_feats, edge_attr, N)

        return node_feats, edge_attr

    def edge_model(self, node_feats, edge_attr):
        f = self.pf_layer(node_feats, edge_attr)
        dE = self.de_mlp(f)
        return edge_attr + dE

    def node_model(self, node_feats, edge_attr, N):
        dV = self.dv_mlp1(edge_attr)
        dV = dV.view(N, N, self.h_nf)
        dV = torch.sum(dV, 1)
        dV = self.dv_mlp2(dV)
        return node_feats + dV


class PairFeaturesLayer(nn.Module):

    def __init__(self, h_nf, act = nn.ReLU()):
        super(PairFeaturesLayer, self).__init__()

        self.h_nf = h_nf
        self.edge_ij_layer = nn.Linear(h_nf, h_nf, bias = True)
        self.node_i_layer = nn.Linear(h_nf, h_nf, bias = False) 
        self.node_j_layer = nn.Linear(h_nf, h_nf, bias = False) 
        self.act = act

    def forward(self, node_feats, edge_attr):
        # edge_attr input dim: N^2xEA

        f_ij = self.edge_ij_layer(edge_attr)
        f_i = self.node_i_layer(node_feats)
        f_j = self.node_j_layer(node_feats)

        # unsqueeze for final addition, then squeeze back again
        N = len(node_feats)
        f_ij = f_ij.view(N, N, self.h_nf)
        f_i = torch.unsqueeze(f_i, 0)
        f_j = torch.unsqueeze(f_j, 1)
        return self.act(f_ij + f_i + f_j).view(N**2, self.h_nf) 


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

    def forward(self, node_feats):
        for i in range(self.num_layers):
            node_out = self.layers[i](node_feats)
        return node_out
        

