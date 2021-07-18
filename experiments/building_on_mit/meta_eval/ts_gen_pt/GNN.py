import torch
import torch.nn as nn

from torch_scatter import scatter_sum

# trying a pytorch version of MIT ts_gen: https://github.com/PattanaikL/ts_gen

class GNN(nn.Module):

    def __init__(self, in_node_nf, in_edge_nf, num_iterations = 3, \
        n_node_layers = 2, h_node_nf = 100, n_edge_layers = 2, h_edge_nf = 100):
        super(GNN, self).__init__()
        # don't need masks because of PyTG batching
        
        self.num_iterations = num_iterations

        # init layers
        self.node_mlp = MLP(in_node_nf, h_node_nf, n_node_layers)
        self.edge_mlp = MLP(in_edge_nf, h_edge_nf, n_edge_layers)
        
        # edge layers 
        out_edge_nf = h_node_nf
        self.pf_layer = PairFeaturesLayer(in_node_nf, in_edge_nf, out_edge_nf)
        self.de_mlp = MLP(out_edge_nf, h_edge_nf, n_edge_layers)

        # node layers
        self.dv_mlp1 = MLP(h_edge_nf, h_node_nf, n_node_layers)
        self.dv_mlp2 = MLP(h_node_nf, h_node_nf, n_node_layers)
        
    
    def forward(self, node_feats, edge_attr, init = True):
        # init hidden states of nodes and edges
        if init:
            node_out = self.node_mlp(node_feats)
            edge_out = self.edge_mlp(edge_attr)

        # iteratively update edges (pair features, MLP, set final), nodes (MLP, reduce, MLP, set final)
        for _ in range(self.num_iterations):
            edge_out = self.edge_model(node_out, edge_out)
            node_out = self.node_model(node_out, edge_out)

        return node_out, edge_out

    def edge_model(self, node_feats, edge_attr):
        f = self.pf_layer(node_feats, edge_attr)
        dE = self.de_mlp(f)
        return edge_attr + dE

    def node_model(self, node_feats, edge_attr):
        dV = self.dv_mlp1(edge_attr)
        dV = torch.sum(dV, dim = 2) # TODO: figure out dim and change in mlp2
        dV = self.dv_mlp2(dV)
        return node_feats + dV


class PairFeaturesLayer(nn.Module):

    def __init__(self, node_nf, edge_nf, out_nf, act = nn.ReLU()):
        super(PairFeaturesLayer, self).__init__()
        
        self.edge_ij_layer = nn.Linear(edge_nf, out_nf, bias = True)
        self.node_i_layer = nn.Linear(node_nf, out_nf, bias = False) # first dim unsqueeze?
        self.node_j_layer = nn.Linear(node_nf, out_nf, bias = False) # second dim unsqueeze?
        self.act = act

    def forward(self, node_feats, edge_attr):
        # lucky uses tf.expand_dims (like unsqueeze) here because of diff node and edge dims
        # not sure if required here
        # NOTE: don't need to get edge_index of features here as assume fully connected graph

        f_ij = self.edge_ij_layer(edge_attr)
        f_i = self.node_i_layer(node_feats)
        f_j = self.node_j_layer(node_feats)

        return self.act(f_ij + f_i + f_j)


class MLP(nn.Module):
    # TODO: add norm layers?

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
        

