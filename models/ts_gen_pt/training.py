import math, os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.utils import to_dense_adj
from models.ts_gen_pt.G2C import G2C
from utils.exp import save_yaml_file, TestLog

MAX_CLIP_NORM = 10


def construct_g2c(dataset, args, device):
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
    
    yaml_file_name = os.path.join(args.log_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, g2c_parameters)

    return g2c, g2c_opt, loss_func


def train(model, loader, loss_func, opt):
    total_loss = 0
    model.train()

    for batch_id, rxn_batch in enumerate(tqdm(loader)):
        
        opt.zero_grad()
        rxn_batch = rxn_batch.to(model.device)
        D_pred, mask, _ = model(rxn_batch)
        D_gt = to_dense_adj(rxn_batch.edge_index, rxn_batch.batch, rxn_batch.y)
        
        batch_loss = loss_func(D_pred, D_gt) / mask.sum()
        batch_loss.backward()
        clip_grad_norm_(model.parameters(), MAX_CLIP_NORM) # clip gradients
#        if logger:
#            pnorm = compute_parameters_norm(model)
#            gnorm = compute_gradients_norm(model)
#            logger.info(f' Batch {batch_id} Loss: {batch_loss.item()}\t Parameter Norm: {pnorm}\t Gradient Norm: {gnorm}')
        opt.step()
        total_loss += batch_loss.item()
    
    RMSE = math.sqrt(total_loss / len(loader.dataset))
    return RMSE


def test(model, loader, loss_func):
    total_loss = 0
    model.eval()
    test_log = TestLog() 
    
    for batch_id, rxn_batch in tqdm(enumerate(loader)):
        rxn_batch = rxn_batch.to(model.device)
        D_pred, mask, W = model(rxn_batch) 
        D_gt = to_dense_adj(rxn_batch.edge_index, rxn_batch.batch, rxn_batch.y)
        
        batch_loss = loss_func(D_pred, D_gt)  / mask.sum()
        total_loss += batch_loss.item()
        test_log.add_D(D_pred.detach().cpu().numpy())
        test_log.add_W(W.detach().cpu().numpy())
    
    RMSE = math.sqrt(total_loss / len(loader.dataset))
    return RMSE, test_log


def compute_parameters_norm(model: nn.Module) -> float:
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))

def compute_gradients_norm(model: nn.Module) -> float:
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))







