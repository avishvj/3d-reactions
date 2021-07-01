import torch.nn as nn
from ..utils import reset

def average(r, p):
    if r.shape != p.shape:
        # this happens on edge features since bonds broken and formed
        # current hacky soln: just average min features
        # TODO: proper solution later
        min_num_fs = min(r.size(0), p.size(0))
        return (r[0:min_num_fs] + p[0:min_num_fs]) / 2
    else:
        return (r + p) / 2

combination_funcs = {'average': average}

class TS_AE(nn.Module):
    """ A graph autoencoder with reactant-product (i.e. two graph) combination logic.
    Encodes to r/p/ts_gt mappings, combines to get ts_premapped/ts_postmapped, decodes to r/p/ts_gt/ts_premap/ts_postmap. 
    Use the three TS mappings/decodings for relative comparison.
    Sources of randomness/variability:
        - NEC layer: coord_edge_mlp: xav_uniform init
        - Training: SGD i.e. batches, other stuff?    
    """

    def __init__(self, encoder, decoder, comb_func = 'average', device = 'cpu'):
        super(TS_AE, self).__init__()

        # architecture
        self.encoder = encoder
        self.decoder = decoder

        # R and P combination function
        assert comb_func in combination_funcs.keys(), f"Combination function not valid. Please pick from {combination_funcs.keys()}"
        # self.comb_func = combination_funcs[self.comb_func]
        self.comb_func = comb_func

        self.to(device)
    
    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)
        
    def forward(self, r_params, p_params, ts_params, batch_node_vecs, premap = True, postmap = True, coord_baseline = False):
        # params = node_feats, edge_index, edge_attr, coords
        # mapped = node_emb, edge_emb, graph_emb, coord_out
        # decoded = recon_node_fs, recon_edge_fs, adj_pred

        # ground truth mols, encoded and decoded
        r_mapped, p_mapped, ts_gt_mapped = self.encode_mols_gt(r_params, p_params, ts_params, batch_node_vecs)
        r_decoded, p_decoded, ts_gt_decoded = self.decode_mols_gt(r_mapped, p_mapped, ts_gt_mapped)

        # record autoencoder pass
        ae_log_dict = {}
        ae_log_dict['r'] = (r_mapped, r_decoded)
        ae_log_dict['p'] = (p_mapped, p_decoded)
        ae_log_dict['ts_gt'] = (ts_gt_mapped, ts_gt_decoded)

        if premap:
            ts_premapped = self.premap(r_params, p_params, batch_node_vecs)
            ts_premap_decoded = self.decoder(ts_premapped)
            ae_log_dict['ts_premap'] = (ts_premapped, ts_premap_decoded)
        
        if postmap:
            ts_postmapped = self.postmap(r_mapped, p_mapped)
            ts_postmap_decoded = self.decoder(ts_postmapped)
            ae_log_dict['ts_postmap'] = (ts_postmapped, ts_postmap_decoded)
        
        if coord_baseline:
            ts_coord_baseline = self.coord_baseline(r_params, p_params)
            ae_log_dict['coord_baseline'] = ts_coord_baseline

        # have three ts mapped: compare in emb space and then decode and compare
        # loss function
        # train on r, p recon + coords OR r, p, ts recon + coords

        # loss functions: emb, coord, recon space

        return ae_log_dict
    
    def combine_r_and_p(self, r_feats, p_feats, comb_func = 'average'):
        assert len(r_feats) == len(p_feats), f"Should map reactant (len: {len(r_feats)}) and \
               product (len: {len(p_feats)}) to same number of features."
        comb_func = combination_funcs[comb_func]
        ts_zip = list(zip(r_feats, p_feats))
        ts_mapped_comb = [comb_func(r_feat, p_feat) for (r_feat, p_feat) in ts_zip]
        return ts_mapped_comb

    def encode_mols_gt(self, r_params, p_params, ts_params, batch_node_vecs):
        r_batch_vec, p_batch_vec, ts_batch_vec = batch_node_vecs
        r_mapped = self.encoder(*r_params, r_batch_vec)
        p_mapped = self.encoder(*p_params, p_batch_vec)
        ts_gt_mapped = self.encoder(*ts_params, ts_batch_vec)
        return r_mapped, p_mapped, ts_gt_mapped
    
    def decode_mols_gt(self, r_mapped, p_mapped, ts_gt_mapped):
        r_decoded = self.decoder(*r_mapped)
        p_decoded = self.decoder(*p_mapped)
        ts_gt_decoded = self.decoder(*ts_gt_mapped)
        return r_decoded, p_decoded, ts_gt_decoded 

    def coord_baseline(self, r_params, p_params):
        _, _, _, r_coords = r_params
        _, _, _, p_coords = p_params
        return self.combine_r_and_p(r_coords, p_coords, 'average')

    def premap(self, r_params, p_params, batch_node_vecs):
        # average r/p raw features as ts params, then encode to ts emb, TODO: r+p batch vec
        r_batch_vec, _, _ = batch_node_vecs
        ts_params_premap = self.combine_r_and_p(r_params, p_params, 'average')
        return self.encoder(*ts_params_premap, r_batch_vec)

    def postmap(self, r_mapped, p_mapped):
        # take mapped r/p raw feats (i.e. r/p embs), and average
        return self.combine_r_and_p(r_mapped, p_mapped, 'average')
    

