import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Embedding

from torch_geometric.nn import radius_graph
from torch_scatter import scatter

from ts_ae.ae import TS_AE


# pytorch version of dimenet++, adapted from dig and source
# https://github.com/klicperajo/dimenet/blob/master/dimenet/model/dimenet_pp.py


class DimeNetPP(nn.Module):

    def __init__(self):
        super(DimeNetPP, self).__init__()

        # cosine basis function expanstion

    
    def forward(self):

        # processing sequence
        #   dist -> rbf
        #   dist+angle -> sbf
        #   
        #   rbf -> emb
        #   rbf + sbf -> interaction
        #   
        #   sum(emb, interactions) -> output mlp

        # functions needed
        #   interatomic dist
        #   neighbour angles from triplets

        return