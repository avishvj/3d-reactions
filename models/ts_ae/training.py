import math
from tqdm import tqdm
from dataclasses import asdict
import torch
import torch.nn.functional as F

from models.ts_ae.tsae import TSAE
from models.encoders.schnet import SchNetEncoder
from models.encoders.egnn import EGNNEncoder

from utils.exp import TestLog
from utils.models import X_to_dist
from torch_geometric.utils import to_dense_batch
from utils.ts_interpolation import SchNetParams, EGNNParams

COORD_DIM = 3

def construct_tsae(dataset, args, device):
    """Create model, optimiser, and loss function."""
    
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
    loss_func = mol_coord_loss 

    return tsae, tsae_opt, loss_func


def mol_coord_loss(D_pred, D_gt):
    """Compare distance matrices of molecule. TODO: might be adj right now -> make IA dist."""
    assert D_pred.shape == D_gt.shape
    return torch.sqrt(F.mse_loss(D_pred, D_gt))

def total_coord_loss(D_preds, D_gts, train_on_ts = False):
    """Compare distance matrices of reaction molecules. """
    r, p, ts = D_preds
    r_gt, p_gt, ts_gt = D_gts

    r_loss = mol_coord_loss(r, r_gt)
    p_loss = mol_coord_loss(p, p_gt)

    if train_on_ts:
        ts_loss = mol_coord_loss(ts, ts_gt)
        return r_loss, p_loss, ts_loss
    
    return r_loss, p_loss


def train(model, loader, loss_func, opt):
    total_loss = 0
    model.train()

    for _, rxn_batch in enumerate(tqdm(loader)):

        opt.zero_grad()
        
        # prepare data for model
        rxn_batch = rxn_batch.to(model.device)
        batch_size = len(rxn_batch.num_atoms)
        max_num_nodes = max(rxn_batch.num_atoms)
        batch_node_vec = rxn_batch.x_r_batch
        r_batch = {'node_feats': rxn_batch.x_r, 'edge_index': rxn_batch.edge_index_r, 'edge_attr': rxn_batch.edge_attr_r, \
            'coords': rxn_batch.pos_r, 'atomic_ns': rxn_batch.z_r, 'batch_node_vec': rxn_batch.x_r_batch}
        p_batch = {'node_feats': rxn_batch.x_p, 'edge_index': rxn_batch.edge_index_p, 'edge_attr': rxn_batch.edge_attr_p, \
            'coords': rxn_batch.pos_p, 'atomic_ns': rxn_batch.z_p, 'batch_node_vec': rxn_batch.x_p_batch}

        # run model
        _, D_pred, _ = model(r_batch, p_batch, max_num_nodes, batch_size, batch_node_vec) 
        
        # create ground truth matrix and calc loss
        X_gt, mask = to_dense_batch(rxn_batch.pos_ts, batch_node_vec, 0., max_num_nodes) # pos_ts = [b * max_num_nodes, 3]
        D_gt = X_to_dist(X_gt)
        batch_loss = loss_func(D_pred, D_gt) / mask.sum()
        
        batch_loss.backward()
        opt.step()
        
        total_loss += batch_loss.item()
    
    RMSE = math.sqrt(total_loss / len(loader.dataset))
    return RMSE


def test(model, loader, loss_func):
    total_loss = 0
    model.eval()
    test_log = TestLog()

    for _, rxn_batch in enumerate(tqdm(loader)):

        # prepare data for model
        rxn_batch = rxn_batch.to(model.device)
        batch_size = len(rxn_batch.num_atoms)
        max_num_nodes = max(rxn_batch.num_atoms)
        batch_node_vec = rxn_batch.x_r_batch
        r_batch = {'node_feats': rxn_batch.x_r, 'edge_index': rxn_batch.edge_index_r, 'edge_attr': rxn_batch.edge_attr_r, \
            'coords': rxn_batch.pos_r, 'atomic_ns': rxn_batch.z_r, 'batch_node_vec': rxn_batch.x_r_batch}
        p_batch = {'node_feats': rxn_batch.x_p, 'edge_index': rxn_batch.edge_index_p, 'edge_attr': rxn_batch.edge_attr_p, \
            'coords': rxn_batch.pos_p, 'atomic_ns': rxn_batch.z_p, 'batch_node_vec': rxn_batch.x_p_batch}

        # run model
        embs, D_pred, _ = model(r_batch, p_batch, max_num_nodes, batch_size, batch_node_vec) 
        
        # create ground truth matrix and calc loss
        X_gt, mask = to_dense_batch(rxn_batch.pos_ts, batch_node_vec, 0., max_num_nodes) # pos_ts = [b * max_num_nodes, 3]
        D_gt = X_to_dist(X_gt)
        batch_loss = loss_func(D_pred, D_gt) / mask.sum()
        total_loss += batch_loss.item()

        # add processing data to log for analysis
        test_log.add_emb(embs)
        test_log.add_D(D_pred) # so can look at PyMol after
    
    RMSE = math.sqrt(total_loss / len(loader.dataset))
    return RMSE, test_log