import math
from tqdm import tqdm
import torch, import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from utils.meta_eval import TestLog # TODO: create unique test log for rep learning


def total_coord_loss(self, pred_coords, gt_coords, max_num_atoms, train_on_ts = False):

    #adj_gt = to_dense_adj(gt_edge_index, max_num_nodes = max_num_atoms).squeeze(dim = 0)
    #assert adj_gt.shape == adj_pred.shape, f"Your adjacency matrices don't have the same shape!" 
    
    r, p, ts = pred_coords
    r_gt, p_gt, ts_gt = gt_coords

    r_loss = torch.sqrt(F.mse_loss(r, r_gt))
    p_loss = torch.sqrt(F.mse_loss(p, p_gt))

    if train_on_ts:
        ts_loss = torch.sqrt(F.mse_loss(ts, ts_gt))
        return r_loss, p_loss, ts_loss
    
    return r_loss, p_loss


def train(model, loader, loss_func, opt):
    total_loss = 0
    model.train()

    for batch_id, rxn_batch in enumerate(tqdm(loader)):

        opt.zero_grad()
        rxn_batch = rxn_batch.to(model.device)
        embs, D_pred = model(rxn_batch) # return mask?
        D_gt = to_dense_adj(rxn_batch.edge_index, rxn_batch.batch, rxn_batch.y)

        batch_loss = loss_func(D_pred, D_gt) # / mask.sum()
        batch_loss.backward()
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
        embs, D_pred = model(rxn_batch) # return mask?
        D_gt = to_dense_adj(rxn_batch.edge_index, rxn_batch.batch, rxn_batch.y)
        
        batch_loss = loss_func(D_pred, D_gt) # / mask.sum()
        total_loss += batch_loss.item()

        # test_log.add_embs(embs)
        # test_log.add_D(D_pred) so can look at PyMol after
    
    RMSE = math.sqrt(total_loss / len(loader.dataset))
    return RMSE, test_log