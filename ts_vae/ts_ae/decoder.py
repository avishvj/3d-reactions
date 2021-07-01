###
import torch
import torch.nn as nn

class MolDecoder(nn.Module):

    def __init__(self, in_node_nf, in_edge_nf, emb_nf, device = 'cpu'):
        super(MolDecoder, self).__init__()
        
        # standard params
        self.in_node_nf = in_node_nf
        self.in_edge_nf = in_edge_nf
        self.emb_nf = emb_nf

        # mlps: node, edge
        self.node_dec_mlp = nn.Linear(emb_nf, in_node_nf)
        self.edge_dec_mlp = nn.Linear(emb_nf, in_edge_nf)
        # decoder adj [found these worked well]
        self.W = nn.Parameter(0.5 * torch.ones(1)).to(device)
        self.b = nn.Parameter(0.8 * torch.ones(1)).to(device)

        self.to(device)
    
    def forward(self, node_emb, edge_emb, graph_emb, coords):
        # i.e. decode
        pass

    def decode(self, node_emb, edge_emb):
        # decode to node_fs, edge_fs, adj
        recon_node_fs = self.node_dec_mlp(node_emb)
        recon_edge_fs = self.edge_dec_mlp(edge_emb)
        adj_pred = self.decode_to_adj(node_emb)
        return recon_node_fs, recon_edge_fs, adj_pred
    
    def decode_to_adj(self):
        pass



