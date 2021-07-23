import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .GNN import GNN, MLP

class G2C(nn.Module):
    """PyTorch version of MIT's ts_gen G2C (graph to coordinates) model https://github.com/PattanaikL/ts_gen. """

    def __init__(self, in_node_nf, in_edge_nf, h_nf, n_layers = 2, num_iterations = 3, device = 'cpu'):
        super(G2C, self).__init__()
        self.gnn = GNN(in_node_nf, in_edge_nf, h_nf, n_layers, num_iterations)
        self.dw_layer = DistWeightLayer(in_nf = h_nf, h_nf = h_nf)
        self.recon = ReconstructCoords(max_dims = 21, coord_dims = 3, total_time = 100)
        self.to(device)

    def forward(self, node_feats, edge_attr):
        gnn_node_out, gnn_edge_out = self.gnn(node_feats, edge_attr)
        D_init, W, emb = self.dw_layer(gnn_edge_out)
        X_pred = self.recon.dist_nlsq(D_init, W)        
        return D_init, W, emb, X_pred

    def rmsd(self, X_pred, X_gt):
        # from https://github.com/charnley/rmsd
        diff = X_pred - X_gt
        num_atoms = len(X_pred)
        return torch.sqrt((diff * diff).sum() / num_atoms)

    def ts_gen_rmsd(self, X_pred, X_gt):
        # reduce same on X1 and X2
        # times masks
        # perturb
        # matmul perturb
        # svd on matmul
        # X1 align
        # calc rmsd

        eps = 1e-2
        X_pred_perturb = X_pred + eps * torch.randn(X_pred.shape)
        X_gt_perturb = X_gt + eps * torch.randn(X_gt.shape)
        A= torch.matmul(X_gt_perturb, X_pred_perturb)
        U, S, V_H = torch.linalg.svd(A, full_matrices = True) # are these same as ts_gen?

        pass    

    def clip_gradients(self):
        pass


class DistWeightLayer(nn.Module):
    
    def __init__(self, in_nf, h_nf, edge_out_nf = 2, n_layers = 1):
        super(DistWeightLayer, self).__init__()
        
        self.edge_out_nf = edge_out_nf
        self.edge_mlp1 = MLP(in_nf, h_nf, n_layers)
        self.edge_mlp2 = nn.Linear(h_nf, edge_out_nf, bias = True)
    
    def forward(self, gnn_edge_out):

        edge_out = self.edge_mlp1(gnn_edge_out)
        emb = torch.sum(edge_out, dim = (0, 1)).unsqueeze(0)
        edge_out = self.edge_mlp2(edge_out) # shape: num_atoms^2 * 2
        # TODO: symmetrise? but shape is irregular? do i need to make num_atoms x num_atoms then add?

        # distance weight predictions
        dist_weight_pred = nn.Softplus()(edge_out.t())
        D_init_vec = dist_weight_pred[0] # NOTE: ignore D_init init here
        W_vec = dist_weight_pred[1]

        # reshaping to matrices, TODO: symmetrise here?
        num_atoms = int(np.sqrt(len(D_init_vec)))
        D_init = D_init_vec.view((num_atoms, num_atoms))
        D_init = D_init * (1 - torch.eye(num_atoms)) # remove self loops
        W = W_vec.view((num_atoms, num_atoms))
        return D_init, W, emb
    

class ReconstructCoords:
    # TODO: remove max_dims as not being used

    def __init__(self, max_dims = 21, coord_dims = 3, total_time = 100):
        # simulation constants
        self.max_dims = max_dims
        self.coord_dims = coord_dims
        self.total_time = total_time

    def dist_nlsq(self, D, W):
        # remove one of the dims used in ts_gen

        # init
        B = self.dist_to_gram(D) # 15x15
        X = self.low_rank_approx_power(B) # want 15x3, currently got 15x15x3
        X += torch.randn(D.shape[0], self.coord_dims) # Nx3

        # opt loop
        t = 0
        while t < self.total_time:
            # t -> t+1, x_i -> x_{i+1}
            t, X = self.step_func(t, X, D, W)
        
        return X
    
    def dist_to_gram(self, D):
        N = len(D)
        D_row = torch.sum(D, 0, keepdim = True) / N
        D_col = torch.sum(D, 1, keepdim = True) / N
        D_mean = torch.sum(D, (0, 1), keepdim = True) / N**2
        B = - 0.5 * (D - D_row - D_col + D_mean)
        return B

    def low_rank_approx_power(self, A, k = 3, num_steps = 10):
        A_lr = A
        u_set = []

        for _ in range(k):    
            # init eigenvector
            u = torch.randn(A.shape[:-1]) # limits between 0 and 1 unlike tf.rand_normal()

            # power iteration
            for _ in range(num_steps):
                u = F.normalize(u, dim = 0, p = 2, eps = 1e-3) # 1 for row, 2 for l2 norm
                u = torch.matmul(A_lr, u)
            
            # rescale by sqrt(eigenvalue)
            eig_sq = torch.sum(torch.square(u), 0, keepdim = True) 
            u = u / torch.pow(eig_sq + 1e-2, 0.25)
            u_set.append(u)
            A_lr = A_lr - torch.matmul(u, u.t())

        X = torch.stack(u_set, -1)
        return X

    def step_func(self, t, X_t, D, W):
        # constants
        tsg_eps1 = 0.1
        tsg_eps2 = 1e-3
        alpha = 5.0
        alpha_base = 0.1
        T = self.total_time

        # init g and dx
        g = self.grad_func(X_t, D, W)[0] # weirdly returns tuple of len 1
        dX = - tsg_eps1 * g

        # speed clipping (how fast in Angstroms)
        speed = torch.sqrt(torch.sum(torch.square(dX), 1, keepdim = True) + tsg_eps2)

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
        tsg_eps = 1e-2
        D_sq = torch.square(torch.unsqueeze(X, 0) - torch.unsqueeze(X, 1))
        D = torch.sum(D_sq, 2) + tsg_eps
        return D


### train and test functions

def train_g2c_epoch(g2c, g2c_opt, loader, test = False):
        # run model, calc loss, opt with adam and clipped grad

        total_loss = 0
        res_dict = {'D_init': [], 'W': [], 'emb': [], 'X_pred': []}

        for batch_id, rxn_batch in enumerate(loader):

            if not test:
                g2c.train()
                g2c_opt.zero_grad()
            else:
                g2c.eval()
            
            # batch
            node_feats, edge_attr = rxn_batch.x, rxn_batch.edge_attr
            # print(f"train node_fs {node_feats.shape}, edge_attr {edge_attr.shape}")
            X_gt = rxn_batch.pos
            batch_vecs = rxn_batch.batch

            # run batch pass of g2c with params
            D_init, W, emb, X_pred = g2c(node_feats, edge_attr)

            batch_loss = g2c.rmsd(X_pred, X_gt)
            total_loss += batch_loss

            if not test:
                batch_loss.backward()
                g2c_opt.step()
            
            # log batch results
            res_dict['D_init'].append(D_init)
            res_dict['W'].append(W)
            res_dict['emb'].append(emb)
            res_dict['X_pred'].append(X_pred)

        return total_loss / batch_id, res_dict