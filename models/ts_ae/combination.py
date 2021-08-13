import torch
import torch.nn as nn

def average(r, p):
    assert r.shape == p.shape, "Reactant and product embeddings are not the same shape!"
    return (r + p) / 2


class Combination(nn.Module):
    
    combination_funcs = {'average': average}

    def __init__(self, comb_func, emb_nf, nn_post = False, nn_comb = False):
        self.cf = self.combination_funcs[comb_func]
        
        # TODO: make mlps
        if nn_post == True:
            self.layer = nn.Linear(emb_nf, emb_nf)  
            self.nn_post = True
        # TODO: make mlps
        if nn_comb == True:
            self.comb_layer = nn.Linear(emb_nf * 2, emb_nf)
    
    def forward(self, r_emb, p_emb):
        
        ts_emb = self.cf(r_emb, p_emb)
        
        if self.use_nn:
            ts_emb = self.layer(ts_emb)

        return ts_emb
    
    def nn_combination(self, r_emb, p_emb):
        rp_embs = torch.cat([r_emb, p_emb])
        ts_emb = self.comb_layer(rp_embs)
        return ts_emb


