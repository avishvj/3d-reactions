# encode r/p, decode r/p/ts
# premap: combine r/p raw features, then map to ts emb, then decode to ts
# postmap: map r and p to r and p embs, then combine r and p embs, then decode to ts
# also encode ts itself separetely on trained version so can compare after

### AE: different concat stages
#   - MolEncoder: NECLayer
#   - MolDecoder: defined in itself
#   - AE itself should be a base class with diff instantiations for feature concat stage
#   - Handle combination logic
# encode R, P, TS separately to get individual embs
# decode them separately to get recon R, P, TS
# in exp/emb_space: have premap and postmap funcs

### sources of randomness
#   - NECLayer: coord_edge_mlp: xav_uniform init


### baselines
# space: embs, coords, recon fs
# type: non-deep (coords), noisy (postmap avg, emb + recon fs), shared decoder ()
# 

# how to do? in training func, create non-deep baseline


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

class TS(nn.Module):
    # simple test

    def __init__(self, encoder, decoder, comb_func = 'average', device = 'cpu'):
        super(TS, self).__init__()

        # architecture
        self.encoder = encoder
        self.decoder = decoder

        # R and P combination function
        assert comb_func in combination_funcs.keys(), f"Combination function not valid. Please pick from {combination_funcs.keys()}"
        # self.comb_func = combination_funcs[self.comb_func]
        self.comb_func = comb_func

        # batch_node_vecs?

        self.to(device)

    def ts_premap(self, r_params, p_params, batch_node_vecs):
        # pre mapping combo: combine, enc, dec TODO: pass in combined r, p batch vec as ts batch vec
        ts_params_premap = self.combine_r_and_p(r_params, p_params)
        ts_mapped_premap = self.encoder(*ts_params_premap, batch_node_vecs[0])
        return ts_mapped_premap
    
    def ts_postmap(self, r_mapped, p_mapped):
        return self.combine_r_and_p(r_mapped, p_mapped)


    def forward(self, r_params, p_params, ts_params, batch_node_vecs):
        # params = node_feats, edge_index, edge_attr, coords
        # mapped = node_emb, edge_emb, graph_emb, coord_out
        
        r_mapped, p_mapped, ts_mapped_gt = self.encode_mols(r_params, p_params, ts_params, batch_node_vecs)
        ts_mapped_premap = self.ts_premap(r_params, p_params)
        ts_mapped_postmap = self.ts_postmap(r_mapped, p_mapped)
        

        # non-deep baseline: average r and p coords
        # noisy baseline: post map average func
        # shared decoder: post map

        # loss function
        # train on r, p recon + coords OR r, p, ts recon + coords

        

        # post mapping combo: encode, combine, decode
        ts_mapped_postmap = self.combine_r_and_p(r_mapped, p_mapped)

        return r_mapped, p_mapped, ts_mapped_gt, ts_mapped_premap, ts_mapped_postmap

    


    def encode_mols(self, r_params, p_params, ts_params, batch_node_vecs):
        
        r_batch_vec, p_batch_vec, ts_batch_vec = batch_node_vecs

        # mapped = node_emb, edge_emb, graph_emb, coord_out
        r_mapped = self.encoder(*r_params, r_batch_vec)
        p_mapped = self.encoder(*p_params, p_batch_vec)
        ts_mapped_gt = self.encoder(*ts_params, ts_batch_vec)

        return r_mapped, p_mapped, ts_mapped_gt
    
    def combine_r_and_p(self, r_feats, p_feats, comb_func = 'average'):
        assert len(r_feats) == len(p_feats), f"Should map reactant (len: {len(r_feats)}) and \
               product (len: {len(p_feats)}) to same number of features."
        comb_func = combination_funcs[comb_func]
        ts_zip = list(zip(r_feats, p_feats))
        ts_mapped_comb = [comb_func(r_feat, p_feat) for (r_feat, p_feat) in ts_zip]
        return ts_mapped_comb
        

    def decode_mols(self, r_mapped, p_mapped, ts_mapped_gt, ts_mapped_comb):
        
        pass

    
    ###


    def full(self, r_params, p_params, ts_params, batch_node_vecs):
        # TODO: where do i need batch node vecs
        # mapped = node_emb, edge_emb, graph_emb, coord_out

        r_vec, p_vec, ts_vec = batch_node_vecs

        # initial mapped
        r_mapped, p_mapped = self.encode_r_and_p(r_params, p_params, batch_node_vecs)

        # baselines
        ts_coord_baseline = self.coord_baseline(r_params, p_params)
        ts_gt_mapped = self.encoder(*ts_params, ts_vec)
        # pre and post map
        ts_premapped = self.premap(r_params, p_params, batch_node_vecs)
        ts_postmapped = self.postmap(r_mapped, p_mapped)

        # decode r, p, ts_gt, ts_premap, ts_postmap 
        r_decoded, p_decoded = self.decode_r_and_p(r_mapped, p_mapped)
        ts_gt_decoded = self.decoder(ts_gt_mapped)
        ts_premap_decoded = self.decoder(ts_premapped)
        ts_postmap_decoded = self.decoder(ts_postmapped)

        # have three ts mapped: compare in emb space and then decode 

        return
        

        
        

    def encode_r_and_p(self, r_params, p_params, batch_node_vecs):
        # mapped = node_emb, edge_emb, graph_emb, coord_out
        r_batch_vec, p_batch_vec, _ = batch_node_vecs
        r_mapped = self.encoder(*r_params, r_batch_vec)
        p_mapped = self.encoder(*p_params, p_batch_vec)
        return r_mapped, p_mapped
    
    def decode_r_and_p(self, r_mapped, p_mapped):
        r_decoded = self.decoder(*r_mapped)
        p_decoded = self.decoder(*p_mapped)
        return r_decoded, p_decoded

    def coord_baseline(self, r_params, p_params):
        _, _, _, r_coords = r_params
        _, _, _, p_coords = p_params
        return self.combine_r_and_p(r_coords, p_coords, 'average')
    
    def premap(self, r_params, p_params, batch_node_vecs):
        # pre mapping combo: combine, enc, dec TODO: pass in combined r, p batch vec as ts batch vec
        r_batch_vec, p_batch_vec, _ = batch_node_vecs
        ts_params_premap = self.combine_r_and_p(r_params, p_params, 'average')
        return self.encoder(*ts_params_premap, r_batch_vec)

    def postmap(self, r_mapped, p_mapped):
        return self.combine_r_and_p(r_mapped, p_mapped, 'average')
    

