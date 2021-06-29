# data processing
from dataclasses import dataclass
from ts_vae.data_processors.grambow_processor import ReactionDataset
from torch_geometric.data import DataLoader
import numpy as np

# my model
from ts_vae.gaes.nec_gae import NodeEdgeCoord_AE
from ts_vae.gaes.ts_creator import TSPostMap

# experiment recording
# from ..exp_utils import BatchLog, EpochLog, ExperimentLog [normal]
from experiments.exp_utils import BatchLog, EpochLog, ExperimentLog # hack for running in notebook
from typing import List
import torch.tensor as Tensor

# plotting
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# torch, torch geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


class Embedding_Exp_Log(ExperimentLog):
    
    def __init__(self, num_rxns, tt_split, batch_size, recorded_batches):
        super(Embedding_Exp_Log, self).__init__(num_rxns, tt_split, batch_size, recorded_batches)

        # embedding save
        self.embeddings = []
    
    def add_embeddings(self, embeddings):
        self.embeddings.append(embeddings)


def ts_interpolation(experiment_params, model_params, loaders):

    # get params out
    num_rxns, tt_split, batch_size, recorded_batches, epochs, test_interval = experiment_params
    nec_ae, nec_opt = model_params
    train_loader, test_loader = loaders

    # recording training
    experiment_log = Embedding_Exp_Log(num_rxns, tt_split, batch_size, recorded_batches)

    for epoch in range(1, epochs):

        # TODO: modify train() to return epoch loss

        train_loss, train_epoch_log, _ = train_nec(nec_ae, nec_opt, train_loader)
        experiment_log.add_epoch(train_epoch_log)
        print(f"===== Training epoch {epoch:03d} complete with loss: {train_loss:.4f} ====")

        if epoch % test_interval == 0:
            # test_loss, test_res = test_nec(nec_ae, test_loader)
            # store r embeddings
            # store best loss
            pass 
    
    # final epoch
    train_loss, train_epoch_log, final_res = train_nec(nec_ae, nec_opt, train_loader)
    experiment_log.add_embeddings(final_res)

    return experiment_log

def print_embeddings(experiment_log):
    # pca on embeddings
    # # embedding: (node_emb, edge_emb, batch)

    # for emb in experiment_log.embeddings:
    #     pass

    node_embs = [node_emb for (node_emb, edge_emb, rxn_batch) in experiment_log.embeddings[0]]
    node_embs = torch.cat(node_embs, dim = 0)
    
    xs, ys = zip(*TSNE().fit_transform(node_embs.detach().numpy()))
    plt.scatter(xs, ys)

    # colour for actual ts vs generated -> i should create embeddings of TS for this
    # color_list = ["green", "purple"]
    # plt.scatter(xs, ys, color = colors)

def train_nec(nec_ae, nec_opt, loader):
    # train model for one epoch
    # create batch logs then create epoch log and return

    epoch_log = EpochLog()
    final_res = []
    total_loss = 0

    for batch_id, rxn_batch in enumerate(loader):

        # train mode + zero gradients
        nec_ae.train()
        nec_opt.zero_grad()

        # init required variables for model
        r_node_feats, r_edge_index, r_edge_attr, r_coords = rxn_batch.x_r, rxn_batch.edge_index_r, rxn_batch.edge_attr_r, rxn_batch.pos_r
        batch_size = len(rxn_batch.idx)
        max_num_atoms = sum(rxn_batch.num_atoms).item() # add this in because sometimes we get hanging atoms if bonds broken

        # run model on reactant
        node_emb, edge_emb, recon_node_fs, recon_edge_fs, adj_pred, coord_out = nec_ae(r_node_feats, r_edge_index, r_edge_attr, r_coords)

        # ground truth values
        ts_node_feats, ts_edge_index, ts_edge_attr, ts_coords = rxn_batch.x_ts, rxn_batch.edge_index_ts, rxn_batch.edge_attr_ts, rxn_batch.pos_ts
        adj_gt = to_dense_adj(ts_edge_index, max_num_nodes = max_num_atoms).squeeze(dim = 0)
        assert adj_gt.shape == adj_pred.shape, f"Your adjacency matrices don't have the same shape! \n \
              GT shape: {adj_gt.shape}, Pred shape: {adj_pred.shape}, Batch size: {batch_size} \n \
              TS edge idx: {ts_edge_index}, TS node_fs shape: {ts_node_feats.shape}, Batch n_atoms: {rxn_batch.num_atoms}"

        # losses and opt step
        adj_loss = F.binary_cross_entropy(adj_pred, adj_gt) # scale: e-1, 10 epochs: 0.5 -> 0.4
        coord_loss = torch.sqrt(F.mse_loss(coord_out, ts_coords)) # scale: e-1, 10 epochs: 1.1 -> 0.4
        node_loss = F.mse_loss(recon_node_fs, ts_node_feats) # scale: e-1 --> e-2, 10 epochs: 0.7 -> 0.05
        batch_loss = adj_loss + coord_loss + node_loss
        total_loss += batch_loss
        batch_loss.backward()
        nec_opt.step()

        # batch logs
        batch_log = BatchLog(batch_size, batch_id, max_num_atoms,
                             coord_loss.item(), adj_loss.item(), node_loss.item(), batch_loss.item())
        epoch_log.add_batch(batch_log)
        batch_res = (node_emb, edge_emb, rxn_batch)
        final_res.append(batch_res)
    
    return total_loss / batch_id, epoch_log, final_res




### with r and p

def train_nec_ts(r_nec_ae, p_nec_ae, r_nec_opt, p_nec_opt, loader):
    # TODO: one ae or two?, maybe need one opt for TS_PostMap rather than two opts for r/p_nec_ae

    epoch_log = EpochLog()
    final_res = []
    total_loss = 0
    
    for batch_id, rxn_batch in enumerate(loader):

        # train mode + zero gradients
        r_nec_ae.train()
        r_nec_opt.zero_grad()
        p_nec_ae.train()
        p_nec_opt.zero_grad()

        # init required variables for model
        r_params = rxn_batch.x_r, rxn_batch.edge_index_r, rxn_batch.edge_attr_r, rxn_batch.pos_r
        p_params = rxn_batch.x_p, rxn_batch.edge_index_p, rxn_batch.edge_attr_p, rxn_batch.pos_p
        batch_size = len(rxn_batch.idx)
        max_num_atoms = sum(rxn_batch.num_atoms).item() # add this in because sometimes we get hanging atoms if bonds broken
        
        # pass params into ts_creator
        ts_creator = TSPostMap('average', r_params, p_params, r_nec_ae, p_nec_ae)
        node_emb, edge_emb, recon_node_fs, recon_edge_fs, adj_pred, coord_out = ts_creator()

        # ground truth values
        ts_node_feats, ts_edge_index, ts_edge_attr, ts_coords = rxn_batch.x_ts, rxn_batch.edge_index_ts, rxn_batch.edge_attr_ts, rxn_batch.pos_ts
        adj_gt = to_dense_adj(ts_edge_index, max_num_nodes = max_num_atoms).squeeze(dim = 0)
        assert adj_gt.shape == adj_pred.shape, f"Your adjacency matrices don't have the same shape! \n \
              GT shape: {adj_gt.shape}, Pred shape: {adj_pred.shape}, Batch size: {batch_size} \n \
              TS edge idx: {ts_edge_index}, TS node_fs shape: {ts_node_feats.shape}, Batch n_atoms: {rxn_batch.num_atoms}"

        # losses and opt step
        adj_loss = F.binary_cross_entropy(adj_pred, adj_gt) # scale: e-1, 10 epochs: 0.5 -> 0.4
        coord_loss = torch.sqrt(F.mse_loss(coord_out, ts_coords)) # scale: e-1, 10 epochs: 1.1 -> 0.4
        node_loss = F.mse_loss(recon_node_fs, ts_node_feats) # scale: e-1 --> e-2, 10 epochs: 0.7 -> 0.05
        batch_loss = adj_loss + coord_loss + node_loss
        total_loss += batch_loss
        batch_loss.backward()
        r_nec_opt.step()
        p_nec_opt.step()

        # batch logs
        batch_log = BatchLog(batch_size, batch_id, max_num_atoms,
                             coord_loss.item(), adj_loss.item(), node_loss.item(), batch_loss.item())
        epoch_log.add_batch(batch_log)
        batch_res = (node_emb, edge_emb, rxn_batch)
        final_res.append(batch_res)
    
    return total_loss / batch_id, epoch_log, final_res

def new_ts_interpolation(experiment_params, model_params, loaders):
    # get params out
    num_rxns, tt_split, batch_size, recorded_batches, epochs, test_interval = experiment_params
    r_nec_ae, r_nec_opt, p_nec_ae, p_nec_opt = model_params
    train_loader, test_loader = loaders

    # log training
    experiment_log = Embedding_Exp_Log(num_rxns, tt_split, batch_size, recorded_batches)

    for epoch in range(1, epochs):
        train_loss, train_epoch_log, _ = train_nec_ts(r_nec_ae, p_nec_ae, r_nec_opt, p_nec_opt, train_loader)
        experiment_log.add_epoch(train_epoch_log)
        print(f"===== Training epoch {epoch:03d} complete with loss: {train_loss:.4f} ====")
    
    # final epoch
    train_loss, train_epoch_log, final_res = train_nec_ts(r_nec_ae, p_nec_ae, r_nec_opt, p_nec_opt, train_loader)
    experiment_log.add_epoch(train_epoch_log)
    experiment_log.add_embeddings(final_res)
    
    return experiment_log

def main_ts():
    torch.set_printoptions(precision = 3, sci_mode = False)

    # data prep
    print("Preparing data...")
    rxns = ReactionDataset(r'data')
    num_rxns = len(rxns)
    train_ratio = 0.8
    num_train = int(np.floor(train_ratio * num_rxns))
    batch_size = 10
    to_follow = ['edge_index_r', 'edge_index_ts', 'edge_index_p', 'edge_attr_r', 'edge_attr_ts', 'edge_attr_p'
                'pos_r', 'pos_ts', 'pos_p', 'x_r', 'x_ts', 'x_p']
    train_loader = DataLoader(rxns[: num_train], batch_size = batch_size, follow_batch = to_follow)
    test_loader = DataLoader(rxns[num_train: ], batch_size = batch_size, follow_batch = to_follow)
    print("Data prepared.\n")

    # model parameters
    print("Preparing models...")
    max_num_atoms = max([rxn.num_atoms.item() for rxn in train_loader.dataset])
    assert all(rxn.num_atom_fs.item() == train_loader.dataset[0].num_atom_fs.item() for rxn in train_loader.dataset)
    num_atom_fs = train_loader.dataset[0].num_atom_fs.item()
    assert all(rxn.num_bond_fs.item() == train_loader.dataset[0].num_bond_fs.item() for rxn in train_loader.dataset)
    num_bond_fs = train_loader.dataset[0].num_bond_fs.item()
    h_nf = 5
    emb_nf = 2

    # models and opts
    r_nec_ae = NodeEdgeCoord_AE(in_node_nf = num_atom_fs, in_edge_nf = num_bond_fs, h_nf = h_nf, out_nf = h_nf, emb_nf = emb_nf)
    r_nec_opt = torch.optim.Adam(r_nec_ae.parameters(), lr = 1e-3)
    p_nec_ae = NodeEdgeCoord_AE(in_node_nf = num_atom_fs, in_edge_nf = num_bond_fs, h_nf = h_nf, out_nf = h_nf, emb_nf = emb_nf)
    p_nec_opt = torch.optim.Adam(p_nec_ae.parameters(), lr = 1e-3)
    print("Models prepared.\n")

    # ts interpolation experiment: train model, get embeddings from train and test
    recorded_batches = []
    epochs = 5
    test_interval = 10
    experiment_params = (num_rxns, train_ratio, batch_size, recorded_batches, epochs, test_interval)
    model_params = (r_nec_ae, r_nec_opt, p_nec_ae, p_nec_opt)
    loaders = (train_loader, test_loader)

    print("Starting TS interpolation experiment...\n")
    experiment_log = new_ts_interpolation(experiment_params, model_params, loaders)
    print("\nCompleted experiment, use the experiment log to print results...")
    # print_embeddings(experiment_log)

    return experiment_log





if __name__ == "__main__":

    # what has the model learned
    #   - fig 3: look at weights corresponding to bonds
    #   - ts interpolation: pca on r and p embeddings

    # initial conditions
    torch.set_printoptions(precision = 3, sci_mode = False)

    # data prep
    print("Preparing data...")
    rxns = ReactionDataset(r'data')
    num_rxns = len(rxns)
    train_ratio = 0.8
    num_train = int(np.floor(train_ratio * num_rxns))
    batch_size = 10
    to_follow = ['edge_index_r', 'edge_index_ts', 'edge_index_p', 'edge_attr_r', 'edge_attr_ts', 'edge_attr_p'
                'pos_r', 'pos_ts', 'pos_p', 'x_r', 'x_ts', 'x_p']
    train_loader = DataLoader(rxns[: num_train], batch_size = batch_size, follow_batch = to_follow)
    test_loader = DataLoader(rxns[num_train: ], batch_size = batch_size, follow_batch = to_follow)
    print("Data prepared.")

    # model parameters
    print("Preparing model...")
    max_num_atoms = max([rxn.num_atoms.item() for rxn in train_loader.dataset])
    assert all(rxn.num_atom_fs.item() == train_loader.dataset[0].num_atom_fs.item() for rxn in train_loader.dataset)
    num_atom_fs = train_loader.dataset[0].num_atom_fs.item()
    assert all(rxn.num_bond_fs.item() == train_loader.dataset[0].num_bond_fs.item() for rxn in train_loader.dataset)
    num_bond_fs = train_loader.dataset[0].num_bond_fs.item()
    h_nf = 5
    emb_nf = 2
    # model and opt
    nec_ae = NodeEdgeCoord_AE(in_node_nf = num_atom_fs, in_edge_nf = num_bond_fs, h_nf = h_nf, out_nf = h_nf, emb_nf = emb_nf)
    nec_opt = torch.optim.Adam(nec_ae.parameters(), lr = 1e-3)
    print("Model prepared.")

    # ts interpolation experiment: train model, get embeddings from train and test
    recorded_batches = []
    epochs = 5
    test_interval = 10
    experiment_params = (num_rxns, train_ratio, batch_size, recorded_batches, epochs, test_interval)
    model_params = (nec_ae, nec_opt)
    loaders = (train_loader, test_loader)
    
    print("Starting TS interpolation experiment...")
    experiment_log = ts_interpolation(experiment_params, model_params, loaders)
    print("\nCompleted experiment, printing results as TSNE plot...")
    print_embeddings(experiment_log)