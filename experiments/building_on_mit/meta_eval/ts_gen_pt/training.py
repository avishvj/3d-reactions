import math, os
from tqdm import tqdm
import torch, torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.utils import to_dense_adj

from new.utils import construct_logger_and_dir, plot_tt_loss, save_yaml_file
from new.data_processing.new_processor import construct_dataset_and_loaders
from new.model.G2C import construct_model_opt_loss
from dataclasses import asdict

MAX_CLIP_NORM = 10

### model training

def train(model, loader, loss_func, opt, logger):
    total_loss = 0
    model.train()

    for batch_id, rxn_batch in enumerate(tqdm(loader)):
        
        opt.zero_grad()
        rxn_batch = rxn_batch.to(model.device)
        D_pred, mask = model(rxn_batch)
        D_gt = to_dense_adj(rxn_batch.edge_index, rxn_batch.batch, rxn_batch.y)
        
        batch_loss = loss_func(D_pred, D_gt) / mask.sum()
        clip_grad_norm_(model.parameters(), MAX_CLIP_NORM) # clip gradients
        if logger:
            pnorm = compute_parameters_norm(model)
            gnorm = compute_gradients_norm(model)
            logger.info(f' Batch {batch_id} Loss: {batch_loss.item()}\t Parameter Norm: {pnorm}\t Gradient Norm: {gnorm}')
        opt.step()
        total_loss += batch_loss.item()
    
    RMSE = math.sqrt(total_loss / len(loader.dataset))
    return RMSE


def test(model, loader, loss_func, log_dir):
    total_loss = 0
    model.eval()
    res_dict = {'D_pred': []} # TODO: add directly into .npy file? using log_dir?
    
    for batch_id, rxn_batch in tqdm(enumerate(loader)):
        rxn_batch = rxn_batch.to(model.device)
        D_pred, mask = model(rxn_batch) 
        D_gt = to_dense_adj(rxn_batch.edge_index, rxn_batch.batch, rxn_batch.y)
        
        batch_loss = loss_func(D_pred, D_gt)  / mask.sum()
        total_loss += batch_loss.item()
        res_dict['D_pred'].append(D_pred.detach().cpu().numpy())
    
    RMSE = math.sqrt(total_loss / len(loader.dataset))
    return RMSE, res_dict


def compute_parameters_norm(model: nn.Module) -> float:
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))

def compute_gradients_norm(model: nn.Module) -> float:
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))







