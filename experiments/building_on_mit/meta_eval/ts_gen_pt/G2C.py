import torch
import torch.nn as nn
import torch.nn.functional as F

from .GNN import GNN, MLP

# pytorch port of MIT ts_gen: https://github.com/PattanaikL/ts_gen

class G2C(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, h_nf, n_layers = 2, num_iterations = 3, device = 'cpu'):
        super(G2C, self).__init__()

        # dims to save
        h_edge_nf = h_nf

        # nn processing
        self.gnn = GNN(in_node_nf, in_edge_nf, num_iterations, node_layers = n_layers, h_node_nf = h_nf, 
            edge_layers = n_layers, h_edge_nf = h_edge_nf)
        self.dw_layer = DistWeightLayer(in_nf = h_edge_nf, h_nf = h_nf)
        self.recon = ReconstructCoords(max_dims = 21, coord_dim = 3, total_time = 100)
        
        self.to(device)

    def forward(self, node_feats, edge_index, edge_attr):
        gnn_node_out, gnn_edge_out = self.gnn(node_feats, edge_index, edge_attr, init = True)
        D_init, W = self.dw_layer(gnn_edge_out)
        X_pred = self.recon.dist_nlsq(D_init, W)        
        return X_pred

    def train(self):
        # TODO: move out of class
        # run layers
        # calc loss
        # optimise with adam and clipped gradients

        # ablation study
        # 1) drop recon_layer
        # 2) do loss with coords instead of dist matrix?
        # 3) use recon (a) init (b) opt
        # 4) change W used in pytorch, pull init out
        # 5) add noise to training coords and run NLS for diff D_inits

        # notes:
        # - is the node update doing anything? yes, it will work for multiple iterations on edge features.
        pass

    def rmsd(self, X1, X2):

        # reduce same on X1 and X2
        # times masks
        # perturb
        # matmul perturb
        # svd on matmul
        # X1 align
        # calc rmsd
        pass    

    def clip_gradients(self):
        pass


class DistWeightLayer(nn.Module):
    
    def __init__(self, in_nf, h_nf, edge_out_nf = 2, n_layers = 1):
        super(DistWeightLayer, self).__init__()

        self.edge_out_nf = edge_out_nf

        # distance pred layers MLP(in_nf, out_nf, n_layers)
        self.edge_mlp1 = MLP(in_nf, h_nf, n_layers)
        self.edge_mlp2 = nn.Linear(h_nf, edge_out_nf, bias = True)
    
    def forward(self, gnn_edge_out):

        edge_out = self.edge_mlp1(gnn_edge_out)
        # squeeze edge_out along cols then rows and save as embedding, TODO: shape of output here
        edge_out = self.edge_mlp2(edge_out)
        edge_out = edge_out + edge_out.t() # symmetrise; last dim of edge out should be matrix of 2-tuples, no batch dim
        
        # some D_init init here?
        D_init = nn.Softplus(edge_out[-1][0])
        D_init = D_init * (1 - torch.eye(len(D_init))) # remove self loops
        # dist_pred = D_init.unsqueeze(3) # should this be 3 here?

        # weights prediction
        W = nn.Softplus(edge_out[-1][1])
    
        return D_init, W # embedding
    

class ReconstructCoords:

    def __init__(self, max_dims = 21, coord_dims = 3, total_time = 100):
        # simulation constants
        self.max_dims = max_dims
        self.coord_dims = coord_dims
        self.total_time = total_time

    def dist_nlsq(self, D, W):
        # init
        B = self.dist_to_gram(D)
        X = self.low_rank_approx_power(B)
        X += torch.randn(D.shape[0], self.max_dims, self.coord_dims) # num_ds, max_node_fs, coord_dims

        # opt loop
        t = 0
        while t < self.total_time:
            # t -> t+1, x_i -> x_{i+1}
            t, X = self.step_func(t, X, D, W)
        
        return X
    
    def dist_to_gram(self, D):
        # each elem of D is real so gram matrix is (D^T)D. normalise?
        return torch.matmul(D.t(), D)

    def low_rank_approx_power(self, A, k = 3, num_steps = 10):
        A_lr = A
        u_set = []        

        for _ in range(k):

            # init eigenvector
            u = torch.unsqueeze(torch.randn(A.shape[:-1]), -1) # this rand might limit to between 0 and 1
            # not sure if same as tf.rand_normal()

            # power iteration
            for _ in range(num_steps):
                u = F.normalize(u, dim = 1, p = 2, eps = 1e-3) # 1 for row, 2 for l2 norm
                u = torch.matmul(A_lr, u)
            
            # rescale by sqrt(eigenvalue)
            eig_sq = torch.sum(torch.square(u), 1)
            u = u / torch.pow(eig_sq + 1e-2, -.25)
            u_set.append(u)
            A_lr = A_lr - torch.matmul(u, u.t())
        
        X = torch.cat(u_set, 2)
        return X

    def step_func(self, t, X_t, D, W):
        # constants
        tsg_eps1 = 0.1
        tsg_eps2 = 1e-3
        alpha = 5.0
        alpha_base = 0.1
        T = self.total_time

        # init g and dx
        g = self.grad_func(X_t, D, W)
        dX = - tsg_eps1 * g

        # speed clipping (how fast in Angstroms)
        speed = torch.sqrt(torch.sum(torch.square(dX), 2), + tsg_eps2)

        # alpha sets max speed (soft trust region)
        alpha_t = alpha_base + (alpha - alpha_base) * ((T - t) / T)
        scale = alpha_t * torch.tanh(speed / alpha_t) / speed
        dX *= scale

        X_new = X_t + dX
        return t+1, X_new
    
    def grad_func(self, X_t, D, W):
        # dist -> energy -> grad
        D_Xt = self.get_euc_dist(X_t)
        U = torch.sum(W * torch.square(D - D_Xt))
        g = torch.autograd.grad(U, X_t)
        return g

    def get_euc_dist(self, X):
        # get euclidean distances
        tsg_eps = 1e-2
        D_sq = torch.square(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2))
        D = torch.sqrt(torch.unsqueeze(D_sq, 3) + tsg_eps) # why unsqueeze 3?
        return D
