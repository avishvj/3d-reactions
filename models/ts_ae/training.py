import math
from models.encoders.schnet import SchNetEncoder
from models.encoders.egnn2 import EGNNEncoder
from models.ts_ae.tsae import TSAE
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from dataclasses import asdict
from utils.exp import TestLog
from utils.models import X_to_dist
from utils.ts_interpolation import SchNetParams, EGNNParams


def construct_tsae(dataset, args, device):
    
    # model
    if args.encoder_type == 'SchNet':
        enc_params = asdict(SchNetParams())
        encoder = SchNetEncoder(**enc_params)
    elif args.encoder_type == 'EGNN':
        # max_num_atoms = max([rxn.num_atoms.item() for rxn in dataset])
        assert all(rxn.num_atom_fs.item() == dataset[0].num_atom_fs.item() for rxn in dataset)
        num_atom_fs = dataset[0].num_atom_fs.item()
        assert all(rxn.num_bond_fs.item() == dataset[0].num_bond_fs.item() for rxn in dataset)
        num_bond_fs = dataset[0].num_bond_fs.item()
        enc_params = asdict(EGNNParams(num_atom_fs, num_bond_fs))
        encoder = EGNNEncoder(**enc_params)
    else:
        raise NotImplementedError(f"Your encoder type {args.encoder_type} is invalid.")

    tsae_parameters = {'encoder': encoder, 'emb_nf': enc_params['h_nf'], 'device': device}
    tsae = TSAE(**tsae_parameters)
    
    # opt and loss
    tsae_opt = torch.optim.Adam(tsae.parameters(), args.lr)
    loss_func = ts_coord_loss # total_coord_loss

    return tsae, tsae_opt, loss_func


def ts_coord_loss(D_pred, D_gt):
    assert D_pred.shape == D_gt.shape
    return torch.sqrt(F.mse_loss(D_pred, D_gt))

def total_coord_loss(pred_coords, gt_coords, train_on_ts = False):

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
        max_num_nodes = max(rxn_batch.num_atoms)
        
        print(rxn_batch)
        r_batch = {'node_feats': rxn_batch.x_r, 'edge_index': rxn_batch.edge_index_r, 'edge_attr': rxn_batch.edge_attr_r, \
            'coords': rxn_batch.pos_r, 'atomic_ns': rxn_batch.z_r, 'batch_node_vec': rxn_batch.x_r_batch}
        p_batch = {'node_feats': rxn_batch.x_p, 'edge_index': rxn_batch.edge_index_p, 'edge_attr': rxn_batch.edge_attr_p, \
            'coords': rxn_batch.pos_p, 'atomic_ns': rxn_batch.z_p, 'batch_node_vec': rxn_batch.x_p_batch}

        embs, D_pred = model(r_batch, p_batch) # return a mask?

        # X = rxn_batch.pos_ts.view(rxn_batch.batch_size, )
        # D_gt = X_to_dist()
        
        D_gt = to_dense_adj(rxn_batch.edge_index_ts, rxn_batch.x_ts_batch, max_num_nodes)
        print(D_gt.shape)

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

        test_log.add_emb(embs)
        test_log.add_D(D_pred) # so can look at PyMol after
    
    RMSE = math.sqrt(total_loss / len(loader.dataset))
    return RMSE, test_log