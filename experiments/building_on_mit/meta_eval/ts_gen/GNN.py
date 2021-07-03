import torch
import torch.nn as nn

from torch_scatter import scatter_sum

# trying a pytorch version of MIT ts_gen: https://github.com/PattanaikL/ts_gen

class GNN(nn.Module):

    def __init__(self, in_node_nf, in_edge_nf, num_epochs = 3, \
        node_layers = 2, h_node_nf = 100, edge_layers = 2, h_edge_nf = 100):
        super(GNN, self).__init__()

        # don't need masks because of PyTG

        self.node_mlp = MLP(in_node_nf, h_node_nf, node_layers)
        self.edge_mlp = MLP(in_edge_nf, h_edge_nf, edge_layers)

        self.num_epochs = num_epochs
        self.inner_layer = PairFeaturesLayer()
    
    def forward(self, node_feats, edge_index, edge_attr):
        
        # init hidden states of nodes and edges
        node_out = self.node_mlp(node_feats)
        edge_out = self.edge_mlp(edge_attr)

        # iteratively update edges (pair features, MLP, set final), nodes (MLP, reduce, MLP, set final)
        for _ in range(self.num_epochs):
            edge_out = self.edge_model(node_out, edge_index, edge_out)
            node_out = self.node_model(node_out, edge_index, edge_out)

        return node_out, edge_out

    def edge_model(self, node_feats, edge_index, edge_attr):
        f = self.pair_features()
        pass

    def node_model(self, node_feats, edge_index, edge_attr):
        pass

    def pair_features(self, node_nf, edge_nf, out_nf, act = nn.ReLU()):
        # lucky uses tf.expand_dims (like unsqueeze) here because of diff node and edge dims
        # not sure if required here

        f_ij = nn.Linear(edge_nf, out_nf, bias = True)
        f_i = nn.Linear(node_nf, out_nf, bias = False) # first dim unsqueeze?
        f_j = nn.Linear(node_nf, out_nf, bias = False) # second dim unsqueeze?

        return act(f_ij + f_i + f_j)

class PairFeaturesLayer(nn.Module):

    def __init__(self, node_nf, edge_nf, out_nf, act = nn.ReLU()):
        super(PairFeaturesLayer, self).__init__()
        
        self.edge_ij_layer = nn.Linear(edge_nf, out_nf, bias = True)
        self.node_i_layer = nn.Linear(node_nf, out_nf, bias = False)
        self.node_j_layer = nn.Linear(node_nf, out_nf, bias = False) 
        self.act = act

    def forward(self, node_feats, edge_index, edge_attr):

        node_is, node_js = edge_index
        node_is_fs, node_js_fs = node_feats[node_is], node_feats[node_js]

        f_ij = self.edge_ij_layer(edge_attr)
        f_i = self.node_i_layer(node_is_fs)
        f_j = self.node_j_layer(node_js_fs)

        return self.act(f_ij + f_i + f_j)

class MLP(nn.Module):
    def __init__(self, in_nf, out_nf, n_layers, act = nn.ReLU()):
        # add norm layers?
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
        
