# data processing
from ts_vae.data_processors.grambow_processor import ReactionDataset
from torch_geometric.data import DataLoader

# torch, torch geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch

# model
from ts_vae.ts_ae.encoder import MolEncoder
from ts_vae.ts_ae.decoder import MolDecoder
from ts_vae.ts_ae.ae import TS_AE, train

# experiment
from experiments.exp_utils import BatchLog, EpochLog, ExperimentLog
import matplotlib.pyplot as plt
import numpy as np

class TSI_ExpLog(ExperimentLog):
    # tsi: transition state interpolation

    def __init__(self, num_rxns, tt_split, batch_size, epochs, test_interval, recorded_batches):
        super(TSI_ExpLog, self).__init__(num_rxns, tt_split, batch_size, epochs, test_interval, recorded_batches)
        self.epoch_ae_results = []
    
    def add_epoch_result(self, res):
        self.epoch_ae_results.append(res)

def tsi_main(tt_split = 0.8, batch_size = 5, epochs = 20, test_interval = 10):

    torch.set_printoptions(precision = 3, sci_mode = False)

    # data prep
    rxns = ReactionDataset(r'data')
    num_rxns = len(rxns)
    num_train = int(np.floor(tt_split * num_rxns))
    to_follow = ['edge_index_r', 'edge_index_ts', 'edge_index_p', 'edge_attr_r', 'edge_attr_ts', 'edge_attr_p'
                'pos_r', 'pos_ts', 'pos_p', 'x_r', 'x_ts', 'x_p']
    train_loader = DataLoader(rxns[: num_train], batch_size = batch_size, follow_batch = to_follow)
    test_loader = DataLoader(rxns[num_train: ], batch_size = batch_size, follow_batch = to_follow)

    # model parameters
    max_num_atoms = max([rxn.num_atoms.item() for rxn in train_loader.dataset])
    assert all(rxn.num_atom_fs.item() == train_loader.dataset[0].num_atom_fs.item() for rxn in train_loader.dataset)
    num_atom_fs = train_loader.dataset[0].num_atom_fs.item()
    assert all(rxn.num_bond_fs.item() == train_loader.dataset[0].num_bond_fs.item() for rxn in train_loader.dataset)
    num_bond_fs = train_loader.dataset[0].num_bond_fs.item()
    h_nf = 5
    emb_nf = 2

    # model and opt
    encoder = MolEncoder(in_node_nf=num_atom_fs, in_edge_nf=num_bond_fs, h_nf=h_nf, out_nf=h_nf, emb_nf=emb_nf)
    decoder = MolDecoder(in_node_nf=num_atom_fs, in_edge_nf=num_bond_fs, emb_nf=emb_nf)
    ts_ae = TS_AE(encoder, decoder)
    ts_opt = torch.optim.Adam(ts_ae.parameters(), lr = 1e-3)

    # tsi experiment: train model, get embs from train and test
    recorded_batches = []
    experiment_params = (num_rxns, tt_split, batch_size, epochs, test_interval, recorded_batches)
    model_params = (ts_ae, ts_opt)
    loaders = (train_loader, test_loader)

    print("Starting TS interpolation experiment...")
    train_log, test_log = tsi(experiment_params, model_params, loaders)
    print("Completed experiment, use the experiment log to print results.")

    return train_log, test_log

def tsi(experiment_params, model_params, loaders):
    
    # get params out
    num_rxns, tt_split, batch_size, epochs, test_interval, recorded_batches = experiment_params
    ts_ae, ts_opt = model_params
    train_loader, test_loader = loaders

    # log training and testing
    train_log = TSI_ExpLog(*experiment_params)
    test_log = TSI_ExpLog(*experiment_params)

    for epoch in range(1, epochs + 1):
        
        train_loss, train_epoch_stats, train_epoch_res = train(ts_ae, ts_opt, train_loader)
        train_log.add_epoch(train_epoch_stats)
        
        if epoch == epochs: # only add final train res
            train_log.add_epoch_result(train_epoch_res)
        
        if epoch % test_interval == 0:
            test_loss, test_epoch_stats, test_epoch_res = train(ts_ae, ts_opt, test_loader, test = True)
            test_log.add_epoch(test_epoch_stats)
            test_log.add_epoch_result(test_epoch_res)
    
    return train_log, test_log

def display_train_and_test_embs(train_log, test_log):
    fig, axs = plt.subplots(1, 2, figsize = (16, 8))
    display_embs(train_log, fig, axs[0])
    display_embs(test_log, fig, axs[1])
    return fig, axs

def display_embs(exp_log, fig, ax):
    # TODO? fig 4: compare test vs train embeddings, plot cosine loss

    # ae_log_dict = {r/p/ts_gt/ts_premap/ts_postmap : (mapped, decoded); batch_node_vecs : batch_node_vecs}
    # mapped = node_emb, edge_emb, graph_emb, coord_out
    # decoded = recon_node_fs, recon_edge_fs, adj_pred

    final_res_batched = exp_log.epoch_ae_results[-1] # = [{batch_res}, {batch_res}, .., {batch_res}]
    final_graph_embs = []
    
    for batch_res in final_res_batched:
        node_embs = batch_res['ts_postmap'][0][0]
        ts_batch_vec = batch_res['batch_node_vecs'][2]
        ts_node_emb_batch = to_dense_batch(node_embs, ts_batch_vec)[0] # [0] cos just append tensors, not true/false values
        
        # graph embs
        r_graph_embs, p_graph_embs, ts_gt_graph_embs = batch_res['r'][0][2], batch_res['p'][0][2], batch_res['ts_gt'][0][2]
        ts_premap_graph_embs, ts_postmap_graph_embs = batch_res['ts_premap'][0][2], batch_res['ts_postmap'][0][2]

        for mol_id, ts_node_emb in enumerate(ts_node_emb_batch):
            graph_emb_tuple = (ts_node_emb, r_graph_embs[mol_id], p_graph_embs[mol_id], ts_gt_graph_embs[mol_id], \
                ts_premap_graph_embs[mol_id], ts_postmap_graph_embs[mol_id])
            final_graph_embs.append(graph_emb_tuple)

    # something simpler? r_graph_embs = [batch_res['r'][0][2] for batch_res in final_res_batched for mol in batch_node_vec]

    r_graph_embs = [r_graph_emb.detach().numpy() for (_, r_graph_emb, _, _, _, _) in final_graph_embs]
    p_graph_embs = [p_graph_emb.detach().numpy() for (_, _, p_graph_emb, _, _, _) in final_graph_embs]
    ts_gt_graph_embs = [ts_gt_graph_emb.detach().numpy() for (_, _, _, ts_gt_graph_emb, _, _) in final_graph_embs]
    ts_premap_graph_embs = [ts_premap_graph_emb.detach().numpy() for (_, _, _, _, ts_premap_graph_emb, _) in final_graph_embs]
    ts_postmap_graph_embs = [ts_postmap_graph_emb.detach().numpy() for (_, _, _, _, _, ts_postmap_graph_emb) in final_graph_embs]

    cols = {'r': 'red', 'p': 'green', 'ts_gt': 'orange', 'ts_premap': 'yellow', 'ts_postmap': 'blue'}

    # fig, ax = plt.subplots(figsize = (8, 8))

    ax.scatter(*zip(*r_graph_embs), color = cols['r'])
    ax.scatter(*zip(*p_graph_embs), color = cols['p'])
    ax.scatter(*zip(*ts_gt_graph_embs), color = cols['ts_gt'])
    ax.scatter(*zip(*ts_premap_graph_embs), color = cols['ts_premap'])
    ax.scatter(*zip(*ts_postmap_graph_embs), color = cols['ts_postmap'])

    markers = [plt.Line2D([0,0], [0,0], color = color, marker = 'o', linestyle = '') for color in  cols.values()]
    ax.legend(markers, cols.keys())

    # title, axes
    num_rxns = exp_log.num_rxns
    train_ratio = exp_log.tt_split
    batch_size = exp_log.batch_size
    epochs = exp_log.get_performed_epochs()
    title = f"{num_rxns} Reactions, Train Ratio: {train_ratio}, {epochs} Epochs, Batch Size: {batch_size}"
    ax.set_title(title)
    ax.set_ylabel('Graph Emb Dim 1')
    ax.set_xlabel('Graph Emb Dim 2')

    return fig, ax

