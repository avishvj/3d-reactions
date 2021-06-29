import torch.nn as nn

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


class TSCreatorBase(nn.Module):
    """ Base class for TS operator. 
    Notes:
        - Two key variables: combination stage and combination function
        - Different concrete subclasses used for different combination stages (pre, post, during mapping).
        - Combination function passed in as string with dictionary mapping to function.
    """

    def __init__(self, comb_func, r_params, p_params, batch_node_vecs):
        super(TSCreatorBase, self).__init__()
        assert comb_func in combination_funcs.keys(), "Combination function not valid."
        self.comb_func = comb_func
        # features for r and p from batch: node_feats, edge_index, edge_attr, coords
        self.r_params = r_params
        self.p_params = p_params
        self.batch_node_vecs = batch_node_vecs
    
    def forward(self):
        return self.create_ts()

    def create_ts(self):
        raise NotImplementedError("Use concrete subclass for different combination stages.")
    
    def combine_r_and_p(self):
        raise NotImplementedError("Use concrete subclass for different combination stages.")

class TSPostMap(TSCreatorBase):
    """ Map the reactant and products then combine their embeddings. 
    Notes:
        - Currently have hacky solution to edge_fs as diff number in R and P as bonds broken/formed.
        - Should work with different feature mappers e.g. EGNN, SchNet, etc.
        - {r/p}_params: node_feats, edge_index, edge_attr, coords
        - {r/p}_mapped: node_emb, edge_emb, recon_node_fs, recon_edge_fs, adj_pred, coord_out
    """

    def __init__(self, comb_func, r_params, p_params, batch_node_vecs, r_mapper, p_mapper):
        super(TSPostMap, self).__init__(comb_func, r_params, p_params, batch_node_vecs)
        # feature mappers e.g. nec_ae
        self.r_mapper = r_mapper
        self.p_mapper = p_mapper
    
    def create_ts(self):
        r_mapped = self.r_mapper(*(self.r_params), self.batch_node_vecs[0])
        p_mapped = self.p_mapper(*(self.p_params), self.batch_node_vecs[1])
        ts_mapped = self.combine_r_and_p(r_mapped, p_mapped)
        return ts_mapped, r_mapped[-1], p_mapped[-1]

    def combine_r_and_p(self, r_mapped, p_mapped):
        """ Combine mapped representations of reactant and product. 
        Notes:
            - As long as mappers give out the same numbers of features in same order, this should work.
        TODO:
            - Check for same ordering of features? This is automatically done if same mapper used for R and P.
        """
        assert len(r_mapped) == len(p_mapped), "Should map reactant and product to same number of features."
        ts_zip = list(zip(r_mapped, p_mapped))
        ts_mapped = [combination_funcs[self.comb_func](r_feat, p_feat) for (r_feat, p_feat) in ts_zip]
        return ts_mapped

class TSPreMap(TSCreatorBase):
    """ Combine the raw reactant and product features to get raw TS, then map to TS embeddings. 
    Notes:
        - Should use this to follow the MIT model more closely.
    """

    def __init__(self, comb_func, r_params, p_params, batch_node_vecs, ts_mapper):
        # batch_node_vecs here is just one vec for ts
        super(TSPreMap, self).__init__(comb_func, r_params, p_params, batch_node_vecs)
        # single ts mapper
        self.ts_mapper = ts_mapper
    
    def create_ts(self):
        ts_params = self.combine_r_and_p()
        ts_mapped = self.ts_mapper(*ts_params, self.batch_node_vecs[0])
        return ts_mapped

    def combine_r_and_p(self):
        ts_zip = list(zip(self.r_params, self.p_params))
        ts_params = [combination_funcs[self.comb_func](r_feat, p_feat) for (r_feat, p_feat) in ts_zip]
        return ts_params






