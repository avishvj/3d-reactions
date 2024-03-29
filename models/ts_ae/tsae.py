import torch
import torch.nn as nn
from models.ts_ae.decoder import TSDecoder

# note: should have util funcs to create GT and LI for tt_split

class TSAE(nn.Module):
    """Autoencoder for creating TS geometries from 2D/3D reactant and product molecular graphs."""

    def __init__(self, encoder, emb_nf, device = 'cpu'):
        super(TSAE, self).__init__()
        self.encoder = encoder # used for reactants and products, all return n emb, g emb, coords
        self.combine = Combination('average', emb_nf, False, device)
        self.decoder = TSDecoder(device)
        self.device = device
        self.to(device)
    
    def forward(self, r_batch, p_batch, max_num_nodes, batch_size, batch_node_vec):
        # convert initial batch data so that R and P have own batches
        # batch data needed: node_feats, edge_index, edge_attr, coords, batch_node_vec, atomic_ns

        # TODO: use node embs in some way?
        r_n_embs, r_g_emb, r_coords = self.encoder(r_batch)
        p_n_embs, p_g_emb, p_coords = self.encoder(p_batch)
        
        ts_n_embs = self.combine(r_n_embs, p_n_embs)
        # ts_g_emb = self.combine(r_g_emb, p_g_emb)

        D_pred, mask = self.decoder(ts_n_embs, max_num_nodes, batch_size, batch_node_vec)
        # D_pred = self.decoder(ts_g_embs)
        
        embs = (r_n_embs, p_n_embs, ts_n_embs)
        # embs = (r_g_emb, p_g_emb, ts_g_emb)

        return embs, D_pred, mask


class Combination(nn.Module):
    """Class to combine reactant and product graph embeddings.
    Includes functions for different combinations.
    """

    def __init__(self, comb_func, emb_nf, nn_for_ts = False, device = 'cpu'):
        super(Combination, self).__init__()
        self.combination_funcs = {'average': self.average, 'nn_comb': self.nn_combination}
        self.cf = self.combination_funcs[comb_func]
        if comb_func == 'nn_comb':
            self.comb_layer = nn.Linear(emb_nf * 2, emb_nf)

        if nn_for_ts == True:
            self.layer = nn.Linear(emb_nf, emb_nf)  
            self.nn_for_ts = True
        else:
            self.nn_for_ts = False

        self.to(device)
    
    def forward(self, r_emb, p_emb):
        assert r_emb.shape == p_emb.shape, "Reactant and product embeddings are not the same shape!"
        ts_emb = self.cf(r_emb, p_emb)
        if self.nn_for_ts:
            ts_emb = self.layer(ts_emb)
        return ts_emb
    
    def nn_combination(self, r_emb, p_emb):
        rp_embs = torch.cat([r_emb, p_emb])
        ts_emb = self.comb_layer(rp_embs)
        return ts_emb
    
    def average(self, r_emb, p_emb):
        return (r_emb + p_emb) / 2


