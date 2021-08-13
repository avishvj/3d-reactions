

from torch.nn import Module

class EGNNEncoder(Module):

    def __init__(self, in_node_nf, in_edge_nf, h_nf, out_nf, emb_nf, act_fn, device):
        super(EGNNEncoder, self).__init__()