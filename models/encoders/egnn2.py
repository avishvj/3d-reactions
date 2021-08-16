import torch
import torch.nn as nn
from torch_geometric.nn.glob.glob import global_mean_pool
from utils.models import unsorted_segment_sum

class EGNNEncoder(nn.Module):
    """PyTorch version of EGNN, mostly lifted from original implementation.
    Sources:
        - EGNN paper: https://arxiv.org/abs/2102.09844
        - https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py
    """

    def __init__(self, in_node_nf, in_edge_nf, h_nf, out_nf, emb_nf, n_layers, act_fn = nn.ReLU(), device = 'cpu'):
        super(EGNNEncoder, self).__init__()

        # main layers (no init layer needed)
        self.n_layers = n_layers
        # TODO: emb init instead?
        self.add_module("EGNN_0", EGNNUpdate(in_node_nf, in_edge_nf, h_nf, out_nf, act_fn)) 
        for l in range(1, n_layers):
            self.add_module(f"EGNN_{l}", EGNNUpdate(out_nf, in_edge_nf, out_nf, out_nf, act_fn))
        
        # final emb processing + create graph emb
        self.post = EGNNPost(out_nf, emb_nf) 
    
        self.to(device)
    
    def forward(self, batch):
        node_feats, batch_node_vec = batch['node_feats'], batch['batch_node_vec']
        edge_index, edge_attr = batch['edge_index'], batch['edge_attr']
        coords = batch['coords']
        for l in range(self.n_layers):
            node_feats, edge_attr, coords = self._modules[f"EGNN_{l}"](node_feats, edge_index, edge_attr, coords)
        node_embs, graph_emb = self.post(node_feats, batch_node_vec)
        return node_embs, graph_emb, coords

### Main classes used for EGNN processing

class EGNNUpdate(nn.Module):
    """Equivariant convolution layer to process nodes, edges, and coordinates.
    Mostly identical to EGNN E_GCL layer: https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py
    """
    def __init__(self, in_node_nf, in_edge_nf, h_nf, out_nf, act_fn = nn.ReLU()):
        super(EGNNUpdate, self).__init__()

        # feat and mlp non-input dims
        out_edge_nf = in_edge_nf
        coord_dim = 3
        radial_dim = 1
        h_coord_nf = radial_dim * 2 # arbitrary, just between num_edge_fs and 1

        # mlp input dims
        in_node_mlp_nf = in_node_nf + out_edge_nf + coord_dim # node_feats + agg + coords
        in_edge_mlp_nf = (in_node_nf * 2) + in_edge_nf + radial_dim
        in_coord_mlp_nf = in_edge_nf # number of edge features

        # mlps: node, edge, coord_edge (no bias, final layer has xav uniform init [following orig]), coord
        self.node_mlp = nn.Sequential(nn.Linear(in_node_mlp_nf, h_nf, True), act_fn, nn.Linear(h_nf, out_nf, True))
        self.edge_mlp = nn.Sequential(nn.Linear(in_edge_mlp_nf, h_nf, True), act_fn, nn.Linear(h_nf, out_edge_nf, True))
        layer = nn.Linear(h_coord_nf, radial_dim, False)
        nn.init.xavier_uniform_(layer.weight, gain = 0.001)
        self.coord_edge_mlp = nn.Sequential(nn.Linear(in_coord_mlp_nf, h_coord_nf), nn.ReLU(), layer)
        self.coord_mlp = nn.Linear(coord_dim, coord_dim)
    
    def forward(self, node_feats, edge_index, edge_attr, coords):
        radial, bond_lengths = self.coord_to_radial(edge_index, coords)
        edge_out = self.edge_update(node_feats, edge_index, edge_attr, radial)
        coord_out = self.coord_update(edge_index, edge_out, coords, bond_lengths)
        node_out = self.node_update(node_feats, edge_index, edge_out, coord_out)
        return node_out, edge_out, coord_out
    
    def coord_to_radial(self, edge_index, coords):
        """Calculate bond lengths and normalise using radial. 
        TODO: Alt coord_norm as class like SE(3).
        """
        atom_is, atom_js = edge_index
        bond_lengths = coords[atom_is] - coords[atom_js]
        radial = torch.sum(bond_lengths**2, 1).unsqueeze(1)
        norm = torch.sqrt(radial) + 1
        normed_bond_lengths = bond_lengths / norm
        return radial, normed_bond_lengths
    
    def edge_update(self, node_feats, edge_index, edge_attr, radial):
        """Create node features for each bonded pair of atoms and run through MLP."""
        atom_is, atom_js = edge_index
        atom_is_fs, atom_js_fs = node_feats[atom_is], node_feats[atom_js]
        edge_in = torch.cat([atom_is_fs, atom_js_fs, edge_attr, radial], dim = 1)
        return self.edge_mlp(edge_in)
    
    def coord_update(self, edge_index, edge_attr, coords, bond_lengths):
        """Update normed bond lengths using epsilon based on bond lengths and MLP(edge). 
        Added Coord_MLP at end to push closer to ground truth.
        """
        atom_is, _ = edge_index
        eps_c = self.coord_edge_mlp(edge_attr) # e_c: ~e-4/e-5
        trans = bond_lengths * eps_c # trans: ~e-5/e-6
        agg = unsorted_segment_sum(trans, atom_is, coords.size(0))
        
        coords += agg
        coord_out = self.coord_mlp(coords)
        return coord_out
    
    def node_update(self, node_feats, edge_index, edge_attr, coords):
        """Using coordinates as feature, doesn't in original."""
        atom_is, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, atom_is, node_feats.size(0))
        node_in = torch.cat([node_feats, agg, coords], dim=1)
        return self.node_mlp(node_in)
    
class EGNNPost(nn.Module):
    """Final EGNN processing for node and graph embeddings."""
    
    def __init__(self, out_nf, emb_nf):
        super(EGNNPost, self).__init__()
        self.node_emb_out = nn.Linear(out_nf, emb_nf)
    
    def forward(self, node_feats, batch_node_vec):
        node_embs = self.node_emb_out(node_feats)
        graph_emb = global_mean_pool(node_embs, batch_node_vec)
        return node_embs, graph_emb





        
