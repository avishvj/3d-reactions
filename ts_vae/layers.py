
import torch
import torch.nn as nn
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)

class GCL(nn.Module):
    # basic graph convolutional layer
    # TODO: make message passing

    # no coord_model -> should maybe have this as an abstract base class to build from
    # or an abstract class under both this basic model and the coord model

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf = 0, act_fn = nn.ReLU(), bias = True):
        super(GCL, self).__init__()
        
        input_edge_nf = input_nf * 2
        self.edge_mlp = nn.Sequential(nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias = bias),
                                      act_fn)
        
        self.node_mlp = nn.Sequential(nn.Linear(hidden_nf + input_nf, hidden_nf, bias = bias),
                                      act_fn)
        
        def edge_model(self, source, target, edge_attr):
            edge_in = torch.cat([source, target], dim = 1)
            if edge_attr is not None:
                edge_in = torch.cat([edge_in, edge_attr], dim = 1)
            out = self.edge_mlp(edge_in)
            return out
        
        def node_model(self, h, edge_index, edge_attr):
            row, col = edge_index
            agg = unsorted_segment_sum(edge_attr, row, num_segments = h.size(0))
            out = torch.cat([h, agg], dim = 1)
            out = self.node_mlp(out)
            return out

def unsorted_segment_sum(edge_attr, segment_ids, num_segments):
    result_shape = (num_segments, edge_attr.size(1))
    result = edge_attr.new_full(result_shape, 0) # init empty result tensor
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, edge_attr) # adds all values from tesnor other int self at indices
    return result


# put this in layers

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class CoordGCLayer(MessagePassing):
    # MP layer
    # all logic takes place in forward() method: loops->transform->normalise->propagate->message
    # propagate() internally calls message(), aggregate(), update() functions

    def __init__(self, in_channels, out_channels):
        super(CoordGCLayer, self).__init__(aggr = 'add')
        self.lin = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]; edge_index has shape [2, E]

        # step 1: add self loops to adj matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        # step 2: transform matrix using networks
        x = self.lin(x)

        # step 3: compute normalisation
        row, col = edge_index
        deg = degree(col, x.size(0), dtype = x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # step 4-5: start propagating messages
        return self.propagate(edge_index, x = x, norm = norm)
    
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # x are node embeddings; norm are norm coefficients

        # step 4: normalise node features
        return norm.view(-1, 1) * x_j