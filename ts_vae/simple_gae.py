import torch
import torch.nn as nn

def unsorted_segment_sum(edge_attr, row, num_segments):
    result_shape = (num_segments, edge_attr.size(1))
    result = edge_attr.new_full(result_shape, 0) # init empty result tensor
    row = row.unsqueeze(-1).expand(-1, edge_attr.size(1))
    result.scatter_add_(0, row, edge_attr) # adds all values from tensor other int self at indices
    return result

# simple node AE

class Node_AE(nn.Module):
    # node now, then layers, then act fn
    # define layers then do
    # then edge, then both
    
    def __init__(self, in_node_nf = 11, in_edge_nf = 4, h_nf = 4, out_nf = 4, emb_nf = 2, device = 'cpu'):
        super(Node_AE, self).__init__()

        self.in_node_nf = in_node_nf
        self.in_edge_nf = in_edge_nf
        self.h_nf = h_nf
        self.out_nf = out_nf
        self.emb_nf = emb_nf
        self.device = device

        # encoder
        nl = Node_Layer(in_nf = in_node_nf + in_edge_nf, h_nf = h_nf, out_nf = out_nf) # , edges_nf = 4) # , act_fn = act_fn)
        self.add_module("Node", nl)
        self.fc_emb = nn.Linear(out_nf, emb_nf) 

        self.to(device)
    
    def forward(self, node_feats, edge_index, edge_attr):
        node_emb = self.encode(node_feats, edge_index, edge_attr)
        adj_pred = self.decode(x = node_emb)
        return adj_pred, node_emb

    def encode(self, node_feats, edge_index, edge_attr):
        # node layer then linear
        node_feats = self._modules["Node"](node_feats, edge_index, edge_attr)
        return self.fc_emb(node_feats)

    def decode(self, x, W = 10, b = -1, remove_diag = True):
        # W, b: weights and biases for linear layer

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


class Node_Layer(nn.Module):

    def __init__(self, in_nf, h_nf, out_nf, bias = True):
        super(Node_Layer, self).__init__()
        # first nn.Linear(in, out): in :- num_node_fs + num_agg_fs == num_node_fs + num_edge_aggr_fs
        self.node_mlp = nn.Sequential(nn.Linear(in_nf, h_nf, bias = bias),
                                      nn.Linear(h_nf, out_nf, bias = bias))
    
    def node_model(self, node_feats, edge_index, edge_attr):
        node_is, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, node_is, node_feats.size(0))
        node_in = torch.cat([node_feats, agg], dim = 1)
        print("node_feats: ", node_feats.shape)
        print("agg: ", agg.shape)
        print("node_in shape: ", node_in.shape)
        print("node mlp: ", self.node_mlp)
        return self.node_mlp(node_in)
    
    def forward(self, node_feats, edge_index, edge_attr):
        node_feats = self.node_model(node_feats, edge_index, edge_attr)
        return node_feats
    
