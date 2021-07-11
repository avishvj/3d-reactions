import torch
import torch.nn as nn
import torch.nn.functional as F

from .GNN import GNN

# pytorch port of MIT ts_gen: https://github.com/PattanaikL/ts_gen

# other notes: adam opt

class G2C:

    def __init__(self, in_node_nf, in_edge_nf, h_nf, n_layers = 2, num_epochs = 3, device = 'cpu'):

        # don't know if needed
        self.dims = {"nodes": in_node_nf, "edges": in_edge_nf}
        self.hps = {"node_layers": n_layers, "node_hidden": h_nf, \
            "edge_layers": n_layers, "edge_hidden": h_nf, "num_epochs": num_epochs}
        
        # this is an autoencoder for coordinates basically
        self.gnn = GNN(in_node_nf, in_edge_nf, num_epochs, node_layers = n_layers, h_node_nf = h_nf, 
            edge_layers = n_layers, h_edge_nf = h_nf)

        self.recon_layer = ReconstructLayer()
        
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

    # need to test these work against tf functions

    def __init__(self):
        pass

    def dither(self):
        pass

    def dist_nlsq(self, D, W, mask):
        # i.e. make this function 'forward' after
        
        # sim constants
        T = 100
        max_dims = 21
        coord_dims = 3

        # init
        B = self.dist_to_gram(D)
        x = self.low_rank_approx_power(B)
        x += torch.randn(D.shape[0], max_dims, coord_dims) # num_ds, max_node_fs, coord_dims

        # opt loop
        t = 0
        while t < T:
            # t -> t+1, x_i -> x_{i+1}
            t, x = self.step_func(t, x, T)
        
        return x

        tf.while_loop(
    cond, body, loop_vars, shape_invariants=None, parallel_iterations=10,
    back_prop=True, swap_memory=False, maximum_iterations=None, name=None
)
    
    def dist_to_gram(self, D):
        # convert dist matrix to gram matrix
        # each elem of D is real so gram matrix is (D^T)D
        # how to deal with batches? should be fine in big adj right?
        # normalise needed?
        return torch.matmul(D.t(), D)

    def low_rank_approx_power(self, A, k = 3, num_steps = 10):

        A_lr = A
        u_set = []
        
        for _ in range(k):

            # init eigenvector
            u = torch.unsqueeze(torch.randn(A.shape[:-1]), -1) # this rand might limit to between 0 and 1
            # not sure if same as tf.rand_normal()

            # power iteration
            for j in range(num_steps):
                u = F.normalize(u, dim = 1, p = 2, eps = 1e-3) # 1 for row, 2 for l2 norm
                u = torch.matmul(A_lr, u)
            
            # rescale by sqrt(eigenvalue)
            eig_sq = torch.sum(torch.square(u), 1)
            u = u / torch.pow(eig_sq + 1e-2, -.25)
            u_set.append(u)
            A_lr = A_lr - torch.matmul(u, u.t())
        
        X = torch.cat(u_set, 2)
        return X

    def step_func(self, t, x_t, T):
        # constants
        tsg_eps1 = 0.1
        tsg_eps2 = 1e-3
        alpha = 5.0
        alpha_base = 0.1

        # init g and dx
        g = self.grad_func(x_t)
        dx = - tsg_eps1 * g

        # speed clipping (how fast in Angstroms)
        speed = torch.sqrt(torch.sum(torch.square(dx), 2), + tsg_eps2)

        # alpha sets max speed (soft trust region)
        alpha_t = alpha_base + (alpha - alpha_base) * ((T - t) / T)
        scale = alpha_t * torch.tanh(speed / alpha_t) / speed
        dx *= scale

        x_new = x_t + dx
        return t+1, x_new
    
    def grad_func(self, D, W, X):
        # dist -> energy -> grad
        D_X = self.get_euc_dist(X)
        U = torch.sum(W * torch.square(D - D_X))
        g = torch.autograd.grad(U, X)
        return g

    def get_euc_dist(self, X):
        # get euclidean distances
        tsg_eps = 1e-2
        D_sq = torch.square(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2))
        D = torch.sqrt(torch.unsqueeze(D_sq, 3) + tsg_eps) # why unsqueeze 3?
        return D

    def rmsd():
        pass

    def clip_gradients(self):
        pass