import torch
import torch.nn as nn

from .GNN import GNN

# pytorch port of MIT ts_gen: https://github.com/PattanaikL/ts_gen

# other notes: adam opt

class G2C:

    def __init__(self, in_node_nf, in_edge_nf, h_nf, n_layers = 2, num_epochs = 3, device = 'cpu'):

        # don't know if needed
        self.dims = {"nodes": in_node_nf, "edges": in_edge_nf}
        self.hps = {"node_layers": n_layers, "node_hidden": h_nf, \
            "edge_layers": n_layers, "edge_hidden": h_nf, "num_epochs": num_epochs}
        
        self.gnn = GNN(in_node_nf, in_edge_nf, num_epochs, node_layers = n_layers, h_node_nf = h_nf, 
            edge_layers = n_layers, h_edge_nf = h_nf)
        
        self.to(device)

    def forward(self):

        # init edge: edge mlp
        # graph pool to get embedding + store [they use sum, not mean]
        # edge out: dense layer
        # set edge
        
        # dist matrix prediction
        # enforce positivity
        # set self-loops = 0
        # store d as d_init

        # weights prediction

        # reconstruct: minimise objective with unrolled gradient descent
        # rmsd loss

        pass

    def node_edge_model(self):
        pass

    def coord_model(self):
        pass

class ReconstructLayer:
    def __init__(self):
        pass

    def dist_to_gram(self):
        pass

    def low_rank_approx(self):
        pass

    def low_rank_approx_power(self):
        pass
    
    def low_rank_approx_weighted(self):
        pass

    def dither(self):
        pass

    def dist_nlsq(self, D, W, mask):
        # i.e. forward
        
        T = 100
        eps = 0.1
        alpha = 5.0
        alpha_base = 0.1
        
        pass
    
    def grad_func(self):
        pass

    def step_func(self):
        pass

    def rmsd():
        pass

    def clip_gradients(self):
        pass

    def distances(self):
        pass