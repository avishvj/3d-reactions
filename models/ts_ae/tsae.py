import torch.nn as nn
from models.ts_ae.combination import Combination
from models.ts_ae.decoder import TSDecoder

# note: should have util funcs to create GT and LI for tt_split

class TSAE(nn.Module):

    def __init__(self, encoder, emb_nf, device = 'cpu'):
        super(TSAE, self).__init__()
        self.encoder = encoder # used for reactants and products, all return node+graph embs
        self.combine = Combination('average', emb_nf, True)
        self.decoder = TSDecoder()
        self.to(device)
    
    def forward(self, r_batch, p_batch):
        
        # encoder
        # get batch node_feats, edge_index, edge_attr, coords
        # create node embs
        # mean pool to get graph emb + node embs
        r_emb = self.encoder(r_batch)
        p_emb = self.encoder(p_batch)
        
        ts_emb = self.combine(r_emb, p_emb)
        D_pred = self.decoder(ts_emb)
        return ts_emb, D_pred


"""
Encoder template: 
init(in_node, in_edge, emb_nf, n_layers, act_fn, device)

They are all the same apart from Layer.So encoder could be the same.
Put SchNet layer stuff into EGNN format: init layer and then actual layer.

Layer needs to follow:
init(emb_nf, n_layers, etc.)
"""



class BaseLayer(nn.Module):
    # based off SphereNet updates

    def __init__(self, in_node_nf = 11, in_edge_nf = 4, h_nf = 4, out_nf = 4, emb_nf = 2, act_fn = nn.ReLU(), device = 'cpu'):        
        super(BaseLayer, self).__init__()
        self.to(device)
    
    def forward(self, batch):
        batch = self.batch_init(batch)
        return self.encode(batch)
    
    def batch_init(self):
        pass

    def encode(self):
        pass

    def node_model(self):
        pass

    def edge_model(self):
        pass

    def graph_model(self):
        pass




