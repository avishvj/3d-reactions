import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .GNN import GNN, MLP

MAX_N = 21
COORD_DIMS = 3
EPS1 = 1e-1
EPS2 = 1e-2
EPS3 = 1e-3
MAX_CLIP_NORM = 10

class G2C(nn.Module):
    """PyTorch version of MIT's ts_gen G2C (graph to coordinates) model https://github.com/PattanaikL/ts_gen.
    Note:
        - Clip gradients function is not needed since torch has a built in version which can be called in the training func directly.
        - For edge_attr, we operate on [batch_size * N * N x num_edge_attr] rather than [batch_size x N x N x num_edge_attr]
          which is why there is some reshaping going on.
        - TODO: check if need to send each layer to device.
    """

    def __init__(self, in_node_nf, in_edge_nf, h_nf, n_layers = 2, gnn_depth = 3, device = 'cpu'):
        super(G2C, self).__init__()
        self.gnn = GNN(in_node_nf, in_edge_nf, h_nf, n_layers, gnn_depth)
        self.dw_layer = DistWeightLayer(in_nf = h_nf, h_nf = h_nf, n_layers = n_layers, device = device)
        self.recon = ReconstructCoords(total_time = 100, device = device)
        self.device = device
        self.to(device)

    def forward(self, node_feats, edge_attr, batch_size, mask_V, mask_E, mask_D):
        gnn_node_out, gnn_edge_out = self.gnn(node_feats, edge_attr, batch_size, mask_V, mask_E)
        D_init, W, emb = self.dw_layer(gnn_edge_out, batch_size, mask_V, mask_D)
        X_pred = self.recon.dist_nlsq(D_init, W, batch_size, mask_D)        
        return D_init, W, emb, X_pred

    def loss_rmsd(self, X_pred, X_gt, mask_temp):
        assert X_pred.shape == X_gt.shape, f"Your coordinate matrices don't match! \
                                            X_pred dim: {X_pred.shape}, X_gt dim: {X_gt.shape}"

        # sum both Xs along max_nodes dim
        X_pred = X_pred - torch.sum(mask_temp * X_pred, 1, keepdim = True) / torch.sum(mask_temp, 1, keepdim = True)
        X_gt = X_gt - torch.sum(mask_temp * X_gt, 1, keepdim = True) / torch.sum(mask_temp, 1, keepdim = True)
        X_pred *= mask_temp
        X_gt *= mask_temp
        
        # add perturbation
        X_pred_perturb = X_pred + EPS2 * torch.randn(X_pred.shape).to(self.device)
        X_gt_perturb = X_gt + EPS2 * torch.randn(X_pred.shape).to(self.device)

        # multiply perturbed and align
        A = torch.matmul(X_gt_perturb.permute(0, 2, 1), X_pred_perturb) 
        U, S, V_H = torch.linalg.svd(A, full_matrices = True) # svdals() instead?
        X_pred_align = torch.matmul(U, torch.matmul(V_H, X_pred.permute(0, 2, 1)))
        X_pred_align = X_pred_align.permute(0, 2, 1)

        # calc metrics
        msd = torch.sum(mask_temp * torch.square(X_pred_align - X_gt), (1, 2))  / torch.sum(mask_temp, (1, 2))
        rmsd = torch.mean(torch.sqrt(msd + EPS3))
        return rmsd, X_pred_align

    def loss_dist_XX(self, X_pred, X_gt, mask_D):
        D_pred = mask_D * self.recon.get_euc_dist(X_pred)
        D_gt = mask_D * self.recon.get_euc_dist(X_gt)
        loss_dist_all = mask_D * torch.abs(D_pred - D_gt)
        loss_dist = torch.sum(loss_dist_all) / torch.sum(mask_D)
        return loss_dist

    def loss_dist_DX(self, D_pred, X_gt, mask_D):
        D_pred = mask_D * D_pred
        D_gt = mask_D * self.recon.get_euc_dist(X_gt)
        loss_dist_all = mask_D * torch.abs(D_pred - D_gt)
        loss_dist = torch.sum(loss_dist_all) / torch.sum(mask_D)
        return loss_dist


class DistWeightLayer(nn.Module):
    
    def __init__(self, in_nf, h_nf, n_layers = 1, edge_out_nf = 2, device = 'cpu'):
        super(DistWeightLayer, self).__init__()
        self.h_nf = h_nf
        self.edge_out_nf = edge_out_nf
        self.edge_mlp1 = MLP(in_nf, h_nf, n_layers)
        self.edge_mlp2 = nn.Linear(h_nf, edge_out_nf, bias = True)
        self.d_init_const = nn.Parameter(torch.tensor(-2.5))
        self.device = device
    
    def forward(self, gnn_edge_out, batch_size, mask_V, mask_D):
        
        # init mlps and embedding
        edge_out = self.edge_mlp1(gnn_edge_out) # TODO: add constant here to increase diffs?
        emb = torch.sum(edge_out.view(batch_size, MAX_N, MAX_N, self.h_nf), dim = (1, 2))
        edge_out = self.edge_mlp2(edge_out).view(batch_size, MAX_N, MAX_N, self.edge_out_nf) 

        # distance weight predictions
        edge_out = edge_out  + torch.transpose(edge_out, 2, 1) # symmetrise
        D_init = nn.Softplus()(self.d_init_const + edge_out[:,:,:, 0]) # dim = batch_size x max_N x max_N
#        D_init = nn.Softplus()(edge_out[:,:,:, 0]) # dim = batch_size x max_N x max_N
        D_init = mask_D * D_init * (1 - torch.eye(MAX_N)).to(self.device) # NOTE: using max_N instead of squeezed mask_V
        W = nn.Softplus()(edge_out[:,:,:, 1])
        return D_init, W, emb
    

class ReconstructCoords(nn.Module):

    def __init__(self, total_time = 100, device = 'cpu'):
        super(ReconstructCoords, self).__init__() 
        # simulation constants
        self.total_time = total_time
        self.alpha_base = nn.Parameter(torch.tensor(0.1))
        self.device = device

    def dist_nlsq(self, D, W, batch_size, mask_D):

        # init
        B = self.dist_to_gram(D, mask_D) 
        X = self.low_rank_approx_power(B) # X dim: bxNx3       

        # opt loop
#        t = 0
#        X += torch.randn(batch_size, MAX_N, COORD_DIMS).to(self.device) # Nx3
#        while t < self.total_time:
#            # t -> t+1, x_i -> x_{i+1}
#            t, X = self.step_func(t, X, D, W, mask_D)
        
        return X
    
    def dist_to_gram(self, D, mask_D):
        D = torch.square(D)
        D_row = torch.sum(D, 1, keepdim = True) / MAX_N
        D_col = torch.sum(D, 2, keepdim = True) / MAX_N
        D_mean = torch.sum(D, (1, 2), keepdim = True) / MAX_N**2
        B = mask_D * -0.5 * (D - D_row - D_col + D_mean)
        return B

    def low_rank_approx_power(self, A, k = 3, num_steps = 10):
        A_lr = A
        u_set = []

        for _ in range(k):

            # init eigenvector
            u = torch.unsqueeze(torch.randn(A.shape[:-1]), -1).to(self.device) # limits between 0 and 1 unlike tf.rand_normal()

            # power iteration
            for _ in range(num_steps):
                u = F.normalize(u, dim = 0, p = 2, eps = EPS3) # 1 for row, 2 for l2 norm
                u = torch.matmul(A_lr, u)
            
            # rescale by sqrt(eigenvalue)
            eig_sq = torch.sum(torch.square(u), 1, keepdim = True) 
            u = u / torch.pow(eig_sq + EPS2, 0.25)
            u_set.append(u)
            A_lr = A_lr - torch.matmul(u, u.permute(0, 2, 1)) # NOTE: double check permute
        
        X = torch.cat(u_set, 2)
        return X

    def step_func(self, t, X_t, D, W, mask_D):
        # constants
        alpha = 5.0
        # alpha_base = 0.1
        T = self.total_time

        # init g and dx
        g = self.grad_func(X_t, D, W, mask_D)[0] # g = tuple(tensor,)[0]
        dX = - EPS1 * g # dX dim: batch_size x max_N x coord_dims

        # speed clipping (how fast in Angstroms)
        speed = torch.sqrt(torch.sum(torch.square(dX), 2, keepdim = True) + EPS3)

        # alpha sets max speed (soft trust region)
        alpha_t = self.alpha_base + (alpha - self.alpha_base) * ((T - t) / T)
        scale = alpha_t * torch.tanh(speed / alpha_t) / speed
        dX *= scale
        X_new = X_t + dX # X_new dim: batch_size x max_N x coord_dims
        return t+1, X_new
    
    def grad_func(self, X_t, D, W, mask_D):
        # dist -> energy -> grad
        # X_t dim: batch_size x max_N x 3
        D_Xt = self.get_euc_dist(X_t) # D_xt dim: batch_size x max_N x max_N
        U = torch.sum(torch.sum(mask_D * W * torch.square(D - D_Xt), (1, 2))) / torch.sum(mask_D, (1, 2)) # U dim: scalar
        U = torch.sum(U)
        g = torch.autograd.grad(U, X_t)
        return g
    
    def get_euc_dist(self, X):
        D_sq = torch.square(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2))
        D = torch.sum(D_sq, 3) + EPS2
        return D


### train and test functions

def train_g2c_epoch(g2c, g2c_opt, loader, test = False):
        # run model, calc loss, opt with adam and clipped grad

        total_loss = 0
        res_dict = {'D_init': [], 'W': [], 'emb': [], 'X_pred': []}

        for batch_id, rxn_batch in enumerate(loader):
            
            # for cuda
            rxn_batch = rxn_batch.to(g2c.device)

            if not test:
                g2c.train()
                g2c_opt.zero_grad()
            else:
                g2c.eval()
            
            # batch
            node_feats, edge_attr = rxn_batch.x, rxn_batch.edge_attr
            batch_size = len(rxn_batch.idx)
            X_gt = rxn_batch.pos.view(batch_size, MAX_N, COORD_DIMS)

            # masks, not sure if these do anything
            mask = sequence_mask(rxn_batch.num_atoms, MAX_N, torch.bool, g2c.device)
            mask_V = mask.view(batch_size * MAX_N, 1)
            mask_temp = torch.unsqueeze(mask, 2)
            mask_E = torch.unsqueeze(mask_temp, 1) * torch.unsqueeze(mask_temp, 2)
            mask_D = torch.squeeze(mask_E, 3)
            mask_E = mask_E.view(batch_size * MAX_N * MAX_N, 1)

            # run batch pass of g2c with params
            D_init, W, emb, X_pred = g2c(node_feats, edge_attr, batch_size, mask_V, mask_E, mask_D)
            batch_loss, _ = g2c.loss_rmsd(X_pred, X_gt, mask_temp)
            # batch_loss = g2c.loss_dist_XX(X_pred, X_gt, mask_D)

            # D loss
#            D_init, W, emb = g2c(node_feats, edge_attr, batch_size, mask_V, mask_E, mask_D)
#            batch_loss = g2c.loss_dist_DX(D_init, X_gt, mask_D)

            total_loss += batch_loss.item()

            if not test:
                batch_loss.backward()
                nn.utils.clip_grad_norm_(g2c.parameters(), MAX_CLIP_NORM)
                g2c_opt.step()
            
            # log batch results
            res_dict['D_init'].append(D_init.detach().cpu().numpy())
            res_dict['W'].append(W.detach().cpu().numpy())
            res_dict['emb'].append(emb.detach().cpu().numpy())
#            res_dict['X_pred'].append(X_pred.detach().cpu().numpy())

        return total_loss / batch_id, res_dict

### masks for pytorch port

def sequence_mask(sizes, max_size = 21, dtype = torch.bool, device = 'cpu'):
    row_vector = torch.arange(0, max_size, 1).to(device)
    matrix = torch.unsqueeze(sizes, dim = -1)
    mask = row_vector < matrix
    mask.type(dtype)
    return mask