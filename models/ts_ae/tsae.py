import math
from platform import node
import torch
import torch.nn.Module as Module

from torch_geometric.nn.glob.glob import global_mean_pool

# note: should have util funcs to create GT and LI for tt_split


class TSAE(Module):

    def __init__(self, encoder, decoder):
        super(TSAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, batch):
        graph_emb = self.encoder(batch)
        D_pred = self.decoder(graph_emb)
        return graph_emb, D_pred
    
    def encode(self, batch):
        pass

    def decode(self, embs):
        pass

    # loss funcs: just for coords (dist matrix)




class RPEncoder_Parent(Module):

    def __init__(self, in_node_nf, in_edge_nf, h_nf, out_nf, emb_nf, device = 'cpu'):
        super(RPEncoder_Parent, self).__init__()

    def forward(self, batch):
        # let's have the embs as 10 dim to start        
        graph_emb = self.encode(batch)
        return graph_emb
    
    def encode(self, batch):
        # get batch node_feats, edge_index, edge_attr, coords
        pass

    
class EGNN_RPEncoder(RPEncoder_Parent):

    def __init__(self, in_node_nf, in_edge_nf, h_nf, out_nf, emb_nf, device):
        super(EGNN_RPEncoder, self).__init__(in_node_nf, in_edge_nf, h_nf, out_nf, emb_nf, device)
    
    def encode(self, batch):
        # get batch node_feats, edge_index, edge_attr, coords
        # create node and edge embs
        # mean pool to get graph emb
        return graph_emb
    

class SchNet_RPEncoder(RPEncoder_Parent):

    def __init__(self, in_node_nf, in_edge_nf, h_nf, out_nf, emb_nf, device):
        super(SchNet_RPEncoder, self).__init__(in_node_nf, in_edge_nf, h_nf, out_nf, emb_nf, device)
    
    def encode(self, batch):
        # get batch node_feats, edge_index, edge_attr, coords
        # create node and edge embs
        # mean pool to get graph emb
        return graph_emb

class DimeNetPP_RPEncoder(RPEncoder_Parent):

    def __init__(self, in_node_nf, in_edge_nf, h_nf, out_nf, emb_nf, device):
        super(EGNN_RPEncoder, self).__init__(in_node_nf, in_edge_nf, h_nf, out_nf, emb_nf, device)
    
    def encode(self, batch):
        # get batch node_feats, edge_index, edge_attr, coords
        # create node and edge embs
        # mean pool to get graph emb
        return graph_emb






class TSDecoder(Module):

    def __init__(self):
        super(TSDecoder, self).__init__()
        pass
    
    def forward(self, embs):
        return self.decode_to_dist(embs)
    
    def decode_to_dist(self, embs):
        # TODO: want to decode to just coordinates i.e. interatomic dist
        # how to decode to coordinates using embs?
        
        node_emb, edge_emb, graph_emb = embs
        
        adj_pred = self.decode_to_adj(node_emb)
        return adj_pred

    def decode_to_adj(self, x, remove_self_loops = True):
            # x dim: [num_nodes, 2], use num_nodes as adj_matrix dim
            # returns probabilistic adj matrix

            # create params from x
            num_nodes = x.size(0)
            x_a = x.unsqueeze(0) # dim: [1, num_nodes, 2]
            x_b = torch.transpose(x_a, 0, 1) # dim: [num_nodes, 1, 2], t.t([_, dim to t, dim to t])

            # generate diffs between node embs as adj matrix
            X = (x_a - x_b) ** 2 # dim: [num_nodes, num_nodes, 2]
            X = X.view(num_nodes ** 2, -1) # dim: [num_nodes^2, 2] to apply sum
            X = torch.sigmoid(self.W * torch.sum(X, dim = 1) + self.b) # sigmoid here since can get negative values with W, b
            # X = torch.tanh(torch.sum(X, dim = 1)) # no linear since can get negative values, gives better output but need diff way of training
            adj_pred = X.view(num_nodes, num_nodes) # dim: [num_nodes, num_nodes] 

            if remove_self_loops:
                adj_pred = adj_pred * (1 - torch.eye(num_nodes))

            return adj_pred


