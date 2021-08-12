import math
import torch
import torch.nn.Module as Module
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj
from utils.meta_eval import TestLog # TODO: create unique test log for rep learning

# note: should have util funcs to create GT and LI for tt_split


class TSAE_Parent(Module):

    def __init__(self):
        super(TSAE_Parent, self).__init__()
    
    def forward(self, batch):
        embs = self.encode(batch)
        D_pred = self.decode(embs)
        return embs, D_pred
    
    def encode(self, batch):
        pass

    def decode(self, embs):
        pass

    def decode_to_adj(self, x, remove_self_loops = True):
        # x dim: [num_nodes, 2], use num_nodes as adj_matrix dim
        # returns probabilistic adj matrix

        # create params from x
        num_nodes = x.size(0)
        x_a = x.unsqueeze(0) # dim: [1, num_nodes, 2]
        x_b = torch.transpose(x_a, 0, 1) # dim: [num_nodes, 1, 2], t.t([_, dim to t, dim to t])

        # generate diffs between node embs as adj matrix
        X = (x_a - x_b) ** 2 # dim: [num_nodes, num_nodes, 2]
        X = X.view(num_nodes ** 2, -1) # dim: [num_nodes^2, 2] to apply sum
        X = torch.sigmoid(self.W * torch.sum(X, dim = 1) + self.b) # sigmoid here since can get negative values with W, b
        # X = torch.tanh(torch.sum(X, dim = 1)) # no linear since can get negative values, gives better output but need diff way of training
        adj_pred = X.view(num_nodes, num_nodes) # dim: [num_nodes, num_nodes] 

        if remove_self_loops:
            adj_pred = adj_pred * (1 - torch.eye(num_nodes))

        return adj_pred


class TSAE(TSAE_Parent):

    def __init__(self):
        super(TSAE, self).__init__()
    
    def forward(self, batch):
        pass

    # loss funcs: just for coords (dist matrix)


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