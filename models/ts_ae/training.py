import math
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj
from utils.meta_eval import TestLog # TODO: create unique test log for rep learning

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