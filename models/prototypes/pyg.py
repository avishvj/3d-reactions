import torch
import torch.nn as nn
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)

# for pyg
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.norm import LayerNorm

### class for coordinate normalisation

class CoordsNorm(nn.Module):
    # special module for coordinate normalisation
    # uses strategy from se3 transformer normalisation
    def __init__(self, eps = 1e-8, scale_init = 1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)
    
    def forward(self, coords):
        norm = coords.norm(dim = -1, keepdim = True)
        normed_coords = coords / norm.clamp(min = self.eps)
        return normed_coords * self.scale

### useful functions

def unsorted_segment_sum(edge_attr, row, num_segments):
    result_shape = (num_segments, edge_attr.size(1))
    result = edge_attr.new_full(result_shape, 0) # init empty result tensor
    row = row.unsqueeze(-1).expand(-1, edge_attr.size(1))
    result.scatter_add_(0, row, edge_attr) # adds all values from tensor other int self at indices
    return result


class GCL_PYG(MessagePassing):
    # MP layer for NEC
    # all logic takes place in forward() method: loops->transform->normalise->propagate->message
    # propagate() internally calls message(), aggregate(), update() functions

    def __init__(self, feats_dim, pos_dim, edge_attr_dim = 0, latent_dim = 2,
                 norm_feats = False, norm_coords = False, norm_coords_scale_init = 1e-2,
                 update_feats = True, update_coords = True,
                 aggr = "add", **kwargs):
        # no soft_edge, fourier_features, dropout, clamp. still don't know what clamp does?
        
        assert aggr in {'add', 'sum', 'max', 'mean'}, "Aggregation method must be valid."
        assert update_feats or update_coords, "You must update features, coordinates, or both."

        super(GCL_PYG, self).__init__(**kwargs)
        
        # model parameters
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.latent_dim = latent_dim
        self.norm_feats = norm_feats
        self.norm_coords = norm_coords
        self.update_feats = update_feats
        self.update_coords = update_coords
        self.edge_input_dim = edge_attr_dim + 1 + (feats_dim * 2) # num edge types + 2*num node feats + 1?

        # edges
        self.edge_mlp = nn.Sequential(nn.Linear(self.edge_input_dim, latent_dim), nn.ReLU())
        self.edge_weight = None # self.edge_weight = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())

        # nodes
        self.node_norm = LayerNorm(feats_dim) if norm_feats else None
        self.coords_norm = CoordsNorm(scale_init = norm_coords_scale_init) if norm_coords else nn.Identity()
        self.node_mlp = nn.Sequential(nn.Linear(feats_dim + latent_dim, feats_dim), nn.ReLU()) if update_feats else None

        # coordinates
        self.coords_mlp = nn.Sequential(nn.Linear(latent_dim, 1)) if update_coords else None

    def forward(self, x, edge_index, edge_attr, batch):

        coords, feats = x[:, : self.pos_dim], x[:, self.pos_dim]

        rel_coords = coords[edge_index[0]] - coords[edge_index[1]]
        rel_dist = (rel_coords ** 2).sum(dim = -1, keepdim = True)

        if edge_attr:
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim = -1)
        else:
            edge_attr_feats = rel_dist
        
        hidden_out, coords_out = self.propagate(edge_index, x = feats, edge_attr = edge_attr_feats,
                            coords = coords, rel_coords = rel_coords, batch = batch)
        
        return torch.cat([coords_out, hidden_out], dim = -1)
    
    def message(self, x_i, x_j, edge_attr):
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr]))
        return m_ij
    
    def propagate(self, edge_index, size, **kwargs):
        
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
        
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)

        # update coordinates if specified
        if self.update_coords:
            coord_wij = self.coords_mlp(m_ij)
            # TODO: clamp if arg set
            # normalise if needed
            kwargs["rel_coords"] = self.coords_norm(kwargs["rel_coords"])

            mhat_i = self.aggregate(coord_wij * kwargs["rel_coords"], **aggr_kwargs)
            coords_out = kwargs["coords"] + mhat_i
        else:
            coords_out = kwargs["coords"]
        
        # update features if specified
        if self.update_feats:
            # TODO: weight the edges if arg passed
            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = self.node_norm(kwargs["x"], kwargs["batch"]) if self.node_norm else kwargs["x"]
            hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim = -1))
            hidden_out = kwargs["x"] + hidden_out
        else:
            hidden_out = kwargs["x"]
        
        # return tuple
        return self.update((hidden_out, coords_out), **update_kwargs)
    
    def __repr__(self):
        dict_print = {}
        return "E(n)-GNN Layer for Graphs: " + str(self.__dict__)