import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, ModuleList, BatchNorm1d
from torch_scatter import scatter_sum

class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, h_nf=100, n_layers=2, gnn_depth=3):
        super(GNN, self).__init__()
        self.n_layers = n_layers
        self.gnn_depth = gnn_depth
        self.node_init = MLP(in_node_nf, h_nf, n_layers)
        self.edge_init = MLP(in_edge_nf, h_nf, n_layers)
        self.update = MetaLayer(NodeModel(h_nf, n_layers), EdgeModel(h_nf, n_layers))
    
    def forward(self, node_feats, edge_index, edge_attr):
        node_feats = self.node_init(node_feats)
        edge_attr = self.edge_init(edge_attr)
        for _ in range(self.gnn_depth):
            node_feats, edge_attr = self.update(node_feats, edge_index, edge_attr)
        return node_feats, edge_attr
    
class MetaLayer(nn.Module):
    # inspired by <https://arxiv.org/abs/1806.01261>`
    def __init__(self, node_model=None, edge_model=None):
        super(MetaLayer, self).__init__()
        self.node_model = node_model
        self.edge_model = edge_model
    
    def forward(self, node_feats, edge_index, edge_attr):
        # NOTE: edge model goes first!
        if self.edge_model is not None:
            edge_attr = edge_attr + self.edge_model(node_feats, edge_index, edge_attr)
        if self.node_model is not None:
            node_feats = node_feats + self.node_model(node_feats, edge_index, edge_attr)
        return node_feats, edge_attr

class NodeModel(nn.Module):
    def __init__(self, h_nf, n_layers):
        super(NodeModel, self).__init__()
        self.node_mlp1 = MLP(h_nf, h_nf, n_layers)
        self.node_mlp2 = MLP(h_nf, h_nf, n_layers)
    
    def forward(self, node_feats, edge_index, edge_attr):
        _, node_js = edge_index
        out = self.node_mlp1(edge_attr)
        out = scatter_sum(out, node_js, dim=0, dim_size=node_feats.size(0))
        return self.node_mlp2(out)

class EdgeModel(nn.Module):
    def __init__(self, h_nf, n_layers):
        super(EdgeModel, self).__init__()
        self.edge_lin = Linear(h_nf, h_nf)
        self.node_in = Linear(h_nf, h_nf, bias=False)
        self.node_out = Linear(h_nf, h_nf, bias=False)
        self.mlp = MLP(h_nf, h_nf, n_layers)
    
    def forward(self, node_feats, edge_index, edge_attr):
        f_ij = self.edge_lin(edge_attr)
        f_i = self.node_in(node_feats)
        f_j = self.node_out(node_feats)
        node_is, node_js = edge_index
        out = F.relu(f_ij + f_i[node_is] + f_j[node_js])
        return self.mlp(out)

class MLP(nn.Module):
    def __init__(self, in_nf, out_nf, n_layers, act=nn.ReLU()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for l in range(n_layers):
            if l == 0:
               self.layers.append(Linear(in_nf, out_nf))
            else:
                self.layers.append(Linear(out_nf, out_nf))
            self.layers.append(nn.Dropout(p=0.5)) # DO
            self.layers.append(nn.Linear(out_nf, out_nf))
            self.layers.append(nn.BatchNorm1d(out_nf))
            self.layers.append(act)
        self.layers.append(nn.Linear(out_nf, out_nf))
    
    def forward(self, feats):
        for l in range(len(self.layers)):
            feats = self.layers[l](feats)
        return feats        


