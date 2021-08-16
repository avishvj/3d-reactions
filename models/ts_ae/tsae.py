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
        
        # convert initial batch data so that R and P have own batches
        # batch data needed: node_feats, edge_index, edge_attr, coords, batch_node_vec, atomic_ns

        # TODO: use node embs
        r_n_embs, r_g_emb = self.encoder(r_batch)
        p_n_embs, p_g_emb = self.encoder(p_batch)
        
        ts_g_emb = self.combine(r_g_emb, p_g_emb)
        embs = (r_g_emb, p_g_emb, ts_g_emb)
        D_pred = self.decoder(ts_g_emb)

        return embs, D_pred


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




