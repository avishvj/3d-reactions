import os, torch, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, Module
from torch_geometric.utils import to_dense_adj

from experiments.exp_utils import save_yaml_file
from experiments.building_on_mit.meta_eval.ts_gen_pt.GNN import GNN, MLP

MAX_N = 21
COORD_DIMS = 3
EPS1 = 1e-1
EPS2 = 1e-2
EPS3 = 1e-3
DW_DIM = 2 # 2 since first dim is dist matrix and second dim is weights matrix

class G2C(Module):
    def __init__(self, in_node_nf, in_edge_nf, h_nf, n_layers=2, gnn_depth=3, device='cpu'):
        super(G2C, self).__init__()
        self.gnn = GNN(in_node_nf, in_edge_nf, h_nf, n_layers, gnn_depth)
        self.edge_mlp = MLP(h_nf, h_nf, n_layers)
        self.pred = Linear(h_nf, DW_DIM) 
        self.act = torch.nn.Softplus()
        self.d_init = torch.nn.Parameter(torch.tensor([4.]), requires_grad=True)
        self.device = device
        # learnable opt params: T=[50]., eps=[0.1], alpha=[5.], alpha_base=[0.1]; add requires_grad=True
        self.to(device)
    
    def forward(self, batch):
        # torch.autograd.set_detect_anomaly(True)   # use only when debugging

        # print(batch.__dict__)

        # create distance and weight matrices predictions
        _, edge_attr = self.gnn(batch.x, batch.edge_index, batch.edge_attr)
        edge_emb = self.edge_mlp(edge_attr)
        dw_pred = self.pred(edge_emb)
        dw_pred = to_dense_adj(batch.edge_index, batch.batch, dw_pred) # shape: bxNxNx2
        dw_pred = dw_pred + dw_pred.permute([0, 2, 1, 3]) # symmetrise

        # use mask and get D+W out
        diag_mask = to_dense_adj(batch.edge_index, batch.batch) # diagonals also masked i.e. 0s on diag
        dw_pred = self.act(self.d_init + dw_pred) * diag_mask.unsqueeze(-1)
        D, W = dw_pred.split(1, dim=-1)
        
        # TODO what is this for?
        n_fill = torch.cat([torch.arange(x) for x in batch.batch.bincount()]) 
        mask = diag_mask.clone()
        mask[batch.batch, n_fill, n_fill] = 1 # fill diags

        # run NLWLS and create final distance matrix
        X_pred = self.dist_nlsq(D.squeeze(-1), W.squeeze(-1), mask)
        batch.coords = X_pred
        D_pred = diag_mask * self.X_to_dist(X_pred)
        return D_pred, diag_mask 

    def dist_nlsq(self, D, W, mask):
        # nonlinear weighted least squares. objective is Sum_ij { w_ij (D_ij - |x_i - x_j|)^2 }
        # shapes: D=bx21x21; W=bx21x21; mask=bx21x21
        T, eps, alpha, alpha_base = 10, EPS1, 5.0, 0.1 # these can be made trainable
        
        def grad_func(X):
            # shapes: X=bx21x3
            X = Variable(X, requires_grad=True) # make X variable to use autograd
            D_X = self.X_to_dist(X) # D_X shape = bx21x21

            # energy calc
            U = torch.sum(mask * W * torch.square(D-D_X), (1,2)) / torch.sum(mask, (1,2))
            U = torch.sum(U) # U now scalar
            # grad calc
            g = torch.autograd.grad(U, X, create_graph=True)[0]
            return g
        
        def step_func(t, X_t):
            # x_t shape = ?x21x3
            g = grad_func(X_t)
            dX = - eps * g 

            # speed clipping (how fast in Angstroms)
            speed = torch.sqrt(torch.sum(torch.square(dX), dim=2, keepdim=True) + EPS3) # bx21x3

            # alpha sets max speed (soft trust region)
            alpha_t = alpha_base + (alpha - alpha_base) * ((T - t) / T)
            scale = alpha_t * torch.tanh(speed / alpha_t) / speed # bx21x1
            dx_scaled = dX * scale # bx21x3

            X_new = X_t + dx_scaled
            return t+1, X_new
        
        B = self.dist_to_gram(D, mask)       # B, D & mask = bx21x21
        X_init = self.low_rank_approx_power(B)   # x_init = bx21x3

        # run simulation
        max_size = D.size(1)
        X_init += torch.normal(0, 1, (D.shape[0], max_size, 3)).to(self.device)
        t = 0
        X = X_init
        while t < T:
            t, X = step_func(t, X)
        
        return X
    
    def dist_to_gram(self, D, mask):
        # Dims: D=bx21x21, mask=bx21x21, n=(b,), n.view([-1,1,1])=bx1x1
        n_atoms_per_mol = mask.sum(dim=1)[:, 0].view(-1, 1, 1)  
        D = torch.square(D)
        D_row = torch.sum(D, dim=1, keepdim=True) / n_atoms_per_mol
        D_col = torch.sum(D, dim=2, keepdim=True) / n_atoms_per_mol
        D_mean = torch.sum(D, dim=[1,2], keepdim=True) / torch.square(n_atoms_per_mol)
        G = mask * -0.5 * (D - D_row - D_col + D_mean)
        return G
    
    def low_rank_approx_power(self, A, k=3, num_steps=10):
        A_lr = A    # A shape (batch, 21, 21)
        u_set = []
        
        for _ in range(k):
            u = torch.unsqueeze(torch.normal(0, 1, A.shape[:-1]), dim=-1).to(self.device) # u shape 1x21x1
            for _ in range(num_steps):
                u = F.normalize(u, dim=1, p=2, eps=EPS3)
                u = torch.matmul(A_lr, u)
            # normalise by scalar value sqrt(eigenvalue)
            eig_sq = torch.sum(torch.square(u), dim=1, keepdim=True)    # eig_sq shape (1,1,1)
            u = u / torch.pow(eig_sq + EPS2, 0.25)   # u shape (batch, 21, 1)
            u_set.append(u)
            # transpose so torch.matmul(u, u.transpose(1,2)) has shape (batch, 21, 21)
            A_lr = A_lr - torch.matmul(u, u.transpose(1,2)) # A_lr shape (batch, 21, 21)
        
        X = torch.cat(tensors=u_set, dim=2)  # X shape (1, 21, 3)
        return X
    
    def X_to_dist(self, X):
        # create euclidean distance matrix from X
        # shapes: X = bx21x3, D = bx21x21
        Dsq = torch.square(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2))
        D = torch.sqrt(torch.sum(Dsq, dim=3) + 1E-2)
        return D


def construct_model_opt_loss(dataset, args, full_log_dir, device):
    # constructs model, optimiser, loss function + saves model params
    # TODO: scheduler here

    # model
    g2c_parameters = {'in_node_nf': dataset.num_node_features, 'in_edge_nf': dataset.num_edge_features,
            'h_nf': args.h_nf, 'gnn_depth': args.gnn_depth, 'n_layers': args.n_layers, 'device': device}
    g2c = G2C(**g2c_parameters)

    # optimiser
    if args.optimiser == 'adam':
        g2c_opt = torch.optim.Adam(g2c.parameters(), args.lr)
    else:
        raise NotImplementedError(f"Optimiser string is invalid. You entered '{args.optimiser}', please select from TODO")
    
    # loss func
    if args.loss == 'mse':
        loss_func = torch.nn.MSELoss(reduction='sum')
    elif args.loss == 'mae':
        loss_func = torch.nn.L1Loss(reduction='sum')
    else:
        raise NotImplementedError(f"Loss function string is invalid. You entered '{args.loss}', please select from TODO")
    
    yaml_file_name = os.path.join(full_log_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, g2c_parameters)

    return g2c, g2c_opt, loss_func