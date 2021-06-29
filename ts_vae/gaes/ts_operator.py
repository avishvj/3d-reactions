# variables: averaging stage, averaging function

# notes:
#   - have to use the same nec_ae throughout
#   - how to deal with diff avg stage? stages 
#       = ['map then avg'=='post_map', 'avg then map'=='pre_map', 'avg during map'=='druing_map']
#   - class for avg funcs
#   - subclasses for different stages?


import torch.nn as nn

def linear_combination(r, p):
    return (r + p) / 2

func_dict = {'linear_combination': linear_combination}

class TS_Operator_Base(nn.Module):
    # pre_map, post_map, during_map subclasses

    def __init__(self, avg_func, r_params, p_params):
        super(TS_Operator_Base, self).__init__()
        assert avg_func in func_dict.keys(), "Averaging function not valid."
        self.avg_func = avg_func
        # features for r and p from batch: node_feats, edge_index, edge_attr, coords
        self.r_params = r_params
        self.p_params = p_params
    
    def forward(self):
        return self.create_ts()

    def create_ts(self):
        raise NotImplementedError("Use concrete subclass for different averaging stages.")
    
    def average_r_and_p(self):
        raise NotImplementedError("Use concrete subclass for different averaging stages.")

class TS_PostMap(TS_Operator_Base):

    def __init__(self, avg_func, r_params, p_params, r_mapper, p_mapper):
        super(TS_PostMap, self).__init__(avg_func, r_params, p_params)
        # feature mappers e.g. nec_ae
        self.r_mapper = r_mapper
        self.p_mapper = p_mapper
    
    def create_ts(self):
        # params: node_feats, edge_index, edge_attr, coords
        # mapped: node_emb, edge_emb, recon_node_fs, recon_edge_fs, adj_pred, coord_out
        r_mapped = self.r_mapper(*(self.r_params))
        p_mapped = self.p_mapper(*(self.p_params))
        ts_mapped = self.average_r_and_p(r_mapped, p_mapped)
        return ts_mapped

    def average_r_and_p(self, r_mapped, p_mapped):
        # mapped: node_emb, edge_emb, recon_node_fs, recon_edge_fs, adj_pred, coord_out
        # as long as mappers give out the same number of items in same order, this should work
        assert len(r_mapped) == len(p_mapped), "Should map reactant and product to same number of features."
        ts_zip = list(zip(r_mapped, p_mapped))
        ts_mapped = [func_dict[self.avg_func](r_feat, p_feat) for (r_feat, p_feat) in ts_zip]
        return ts_mapped

class TS_PreMap(TS_Operator_Base):
    # this should follow the MIT model more closely

    def __init__(self, avg_func, r_params, p_params, ts_mapper):
        super(TS_PreMap, self).__init__(avg_func, r_params, p_params)
        # single ts mapper
        self.ts_mapper = ts_mapper
    
    def create_ts(self):
        # params: node_feats, edge_index, edge_attr, coords
        # mapped: node_emb, edge_emb, recon_node_fs, recon_edge_fs, adj_pred, coord_out
        ts_params = self.average_r_and_p()
        ts_mapped = self.ts_mapper(*ts_params)
        return ts_mapped

    def average_r_and_p(self):
        # use avg_func to combine r and p beforehand
        ts_zip = list(zip(self.r_params, self.p_params))
        ts_params = [func_dict[self.avg_func](r_feat, p_feat) for (r_feat, p_feat) in ts_zip]
        return ts_params






