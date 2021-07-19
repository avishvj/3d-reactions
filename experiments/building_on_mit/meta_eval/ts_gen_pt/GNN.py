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
        h_nf = h_node_nf # TODO: remove node and edge h_nf; they have to be same because add in pf_layer
        self.pf_layer = PairFeaturesLayer(h_nf)
        self.de_mlp = MLP(h_nf, h_nf, n_edge_layers) 

        # node layers
        self.dv_mlp1 = MLP(h_edge_nf, h_node_nf, n_node_layers)
        self.dv_mlp2 = MLP(h_node_nf, h_node_nf, n_node_layers)
        
    
    def forward(self, node_feats, edge_attr, init = True):
        # init hidden states of nodes and edges
        if init:
            node_feats = self.node_mlp(node_feats)
            edge_attr = self.edge_mlp(edge_attr)
            # TODO: if not init, then need different dimensions in following mlps here!
        
        print(f"gnn fwd: nf {node_feats.shape}, ea {edge_attr.shape}")

        # iteratively update edges (pair features, MLP, set final), nodes (MLP, reduce, MLP, set final)
        for _ in range(self.num_iterations):
            edge_attr = self.edge_model(node_feats, edge_attr)
            node_feats = self.node_model(node_feats, edge_attr)

        return node_feats, edge_attr

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

    def __init__(self, h_nf, act = nn.ReLU()):
        super(PairFeaturesLayer, self).__init__()
        
        self.h_nf = h_nf
        self.edge_ij_layer = nn.Linear(h_nf, h_nf, bias = True)
        self.node_i_layer = nn.Linear(h_nf, h_nf, bias = False) # first dim unsqueeze?
        self.node_j_layer = nn.Linear(h_nf, h_nf, bias = False) # second dim unsqueeze?
        self.act = act

    def forward(self, node_feats, edge_attr):
        # lucky uses tf.expand_dims (like unsqueeze) here because of diff node and edge dims
        # not sure if required here
        # NOTE: don't need to get edge_index of features here as assume fully connected graph

        print(f"pf layer fwd: nf {node_feats.shape}, ea {edge_attr.shape}")

        f_ij = self.edge_ij_layer(edge_attr)
        f_i = self.node_i_layer(node_feats)
        f_j = self.node_j_layer(node_feats)

        num_atoms = f_i.shape[0]
        assert num_atoms**2 == f_ij.shape[0], "Number of nodes^2 not equal to number of edge attr." 
        # TODO: sort out when batch_size > 1
        fi_fj = torch.randn(num_atoms**2, self.h_nf)
        for i in range(num_atoms):
            for j in range(num_atoms):
                fi_fj[i * num_atoms + j] = torch.matmul(f_i[i], f_j[j]) 

        # return self.act(f_ij + f_i + f_j)
        return self.act(f_ij + fi_fj)


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
        

