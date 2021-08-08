import math
from tqdm import tqdm
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.utils import to_dense_adj
from utils.meta_eval import TestLog

MAX_CLIP_NORM = 10

def train(model, loader, loss_func, opt, logger):
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







