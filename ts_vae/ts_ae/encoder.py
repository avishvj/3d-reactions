import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.glob.glob import global_mean_pool

from ..utils import unsorted_segment_sum

class MolEncoder(nn.Module):
    """ Encoder class that uses NECLayer. """

    def __init__(self, in_node_nf = 11, in_edge_nf = 4, h_nf = 4, out_nf = 4, emb_nf = 2, act_fn = nn.ReLU(), device = 'cpu'):
        super(MolEncoder, self).__init__()

        # standard params
        self.in_node_nf = in_node_nf
        self.in_edge_nf = in_edge_nf
        self.h_nf = h_nf
        self.out_nf = out_nf
        self.emb_nf = emb_nf
        self.device = device
        # coord params?

        # main layer(s)
        necl = NECLayer(in_node_nf, in_edge_nf, h_nf, out_nf, act_fn)
        self.add_module("NEC", necl)

        # embedding creation
        self.node_emb_mlp = nn.Linear(out_nf, emb_nf)
        out_edge_nf = in_edge_nf
        self.edge_emb_mlp = nn.Linear(out_edge_nf, emb_nf)

        self.to(device)
    
    def forward(self, node_feats, edge_index, edge_attr, coords, node_batch_vec):
        node_emb, edge_emb, coord_out = self.encode(node_feats, edge_index, edge_attr, coords)
        graph_emb = global_mean_pool(node_emb, node_batch_vec)
        return node_emb, edge_emb, graph_emb, coord_out

    def encode(self, node_feats, edge_index, edge_attr, coords):
        node_out, edge_out, coord_out = self._modules["NEC"](node_feats, edge_index, edge_attr, coords)
        node_emb = self.node_emb_mlp(node_out)
        edge_emb = self.edge_emb_mlp(edge_out)
        return node_emb, edge_emb, coord_out

class NECLayer(nn.Module):
    """ Layer to process nodes, edges, and coordinates. Based off EGNN. """

    def __init__(self, in_node_nf, in_edge_nf, h_nf, out_nf, bias = True, act_fn = nn.ReLU()):
        super(NECLayer, self).__init__()
        # node_norm : LayerNorm; coord_norm: define (follow se3); edge_norm: required?

        # setting feat and mlp non-input dims
        out_edge_nf = in_edge_nf
        coord_dim = 3
        radial_dim = 1
        h_coord_nf = radial_dim * 2 # arbitrary, just between num_edge_fs and 1

        # mlp input dims
        in_node_mlp_nf = in_node_nf + out_edge_nf + coord_dim # node_feats + agg + coords
        in_edge_mlp_nf = (in_node_nf * 2) + in_edge_nf + radial_dim
        in_coord_mlp_nf = in_edge_nf # number of edge features

        # mlps: node, edge, coord_edge (no bias, final layer has xav uniform init [following orig]), coord
        self.node_mlp = nn.Sequential(nn.Linear(in_node_mlp_nf, h_nf, bias), act_fn, nn.Linear(h_nf, out_nf, bias))
        self.edge_mlp = nn.Sequential(nn.Linear(in_edge_mlp_nf, h_nf, bias), act_fn, nn.Linear(h_nf, out_edge_nf, bias))
        layer = nn.Linear(h_coord_nf, radial_dim, False)
        torch.nn.init.xavier_uniform_(layer.weight, gain = 0.001)
        self.coord_edge_mlp = nn.Sequential(nn.Linear(in_coord_mlp_nf, h_coord_nf), nn.ReLU(), layer)
        self.coord_mlp = nn.Linear(coord_dim, coord_dim)
    
    def forward(self, node_feats, edge_index, edge_attr, coords):
        # TODO: coords at end so can use node_feats
        radial, bond_lengths = self.coord_to_radial(edge_index, coords)
        edge_out = self.edge_model(node_feats, edge_index, edge_attr, radial)
        coord_out = self.coord_model(edge_index, edge_out, coords, bond_lengths)
        node_out = self.node_model(node_feats, edge_index, edge_out, coord_out)
        return node_out, edge_out, coord_out

    def coord_to_radial(self, edge_index, coords):
        # calc bond lengths
        atom_is, atom_js = edge_index
        bond_lengths = coords[atom_is] - coords[atom_js]
        # normalise coords, TODO: alternative coord_norm as func or class
        radial = torch.sum((bond_lengths)**2, 1).unsqueeze(1)
        norm = torch.sqrt(radial) + 1
        normed_bond_lengths = bond_lengths / norm
        return radial, normed_bond_lengths

    def node_model(self, node_feats, edge_index, edge_attr, coords):
        # NOTE: edge_attr is edge_out; currently using coords as feature, doesn't in original
        node_is, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, node_is, node_feats.size(0))
        node_in = torch.cat([node_feats, agg, coords], dim = 1)
        return self.node_mlp(node_in)
    
    def edge_model(self, node_feats, edge_index, edge_attr, radial):
        node_is, node_js = edge_index
        # get node feats for each bonded pair of atoms
        node_is_fs, node_js_fs = node_feats[node_is], node_feats[node_js]
        edge_in = torch.cat([node_is_fs, node_js_fs, edge_attr, radial], dim = 1)
        return self.edge_mlp(edge_in)
    
    def coord_model(self, edge_index, edge_attr, coords, bond_lengths):
        # radially normed coord differences * coord_mlp(edges) == normed bond lengths * edge_out
        # NOTE: edge_attr here is edge_out; added coord_mlp to push coords closer to final value
        # TODO: eval impact of coord_edge_mlp(), suspicion it does nothing as edge not too relevant
        # TODO: is equivariance preserved with coord_mlp()?
        atom_is, _ = edge_index
        e_c = self.coord_edge_mlp(edge_attr) # e_c: ~e-4/e-5
        trans = bond_lengths * e_c   # trans: ~e-5/e-6
        agg = unsorted_segment_sum(trans, atom_is, coords.size(0))
        coords += agg
        coord_out = self.coord_mlp(coords)
        return coord_out
    



