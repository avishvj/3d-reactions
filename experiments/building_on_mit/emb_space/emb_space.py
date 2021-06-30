# data processing
from ts_vae.data_processors.grambow_processor import ReactionDataset
from torch_geometric.data import DataLoader
import numpy as np

# my model
from ts_vae.gaes.nec_gae import NodeEdgeCoord_AE
from ts_vae.gaes.ts_creator import TSPostMap

# experiment recording
# from ..exp_utils import BatchLog, EpochLog, ExperimentLog [normal]
from experiments.exp_utils import BatchLog, EpochLog, ExperimentLog # hack for running in notebook

# plotting
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# torch, torch geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch


class Embedding_Exp_Log(ExperimentLog):
    
    def __init__(self, num_rxns, tt_split, batch_size, epochs, recorded_batches):
        super(Embedding_Exp_Log, self).__init__(num_rxns, tt_split, batch_size, epochs, recorded_batches)
        # embedding save
        self.embeddings = []
    
    def add_embs_and_batch(self, embeddings):
        # each embedding is (node_emb, edge_emb, rxn_batch)
        self.embeddings.append(embeddings)

def train_tsi(r_nec_ae, p_nec_ae, r_nec_opt, p_nec_opt, loader):
    # TODO: one ae or two?, maybe need one opt for TS_PostMap rather than two opts for r/p_nec_ae

    epoch_log = EpochLog()
    final_embs = []
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
        batch_node_vecs = rxn_batch.x_r_batch, rxn_batch.x_p_batch # for recreating graphs, TODO: {x_r/x_p/x_ts}_batch?
        
        # pass params into ts_creator and get ts feats
        ts_creator = TSPostMap('average', r_params, p_params, batch_node_vecs, r_nec_ae, p_nec_ae)
        # node_emb, edge_emb, recon_node_fs, recon_edge_fs, adj_pred, coord_out, ts_graph_emb, r_graph = ts_creator()
        
        ts_mapped, r_graph_emb, p_graph_emb = ts_creator()
        node_emb, edge_emb, recon_node_fs, recon_edge_fs, adj_pred, coord_out, ts_graph_emb = ts_mapped

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

        # log batch results
        batch_log = BatchLog(batch_size, batch_id, max_num_atoms, batch_node_vecs,
                             coord_loss.item(), adj_loss.item(), node_loss.item(), batch_loss.item())
        epoch_log.add_batch(batch_log)
        batch_embs = (node_emb, ts_graph_emb, rxn_batch, r_graph_emb, p_graph_emb)
        final_embs.append(batch_embs)
    
    return total_loss / batch_id, epoch_log, final_embs

def ts_interpolation(experiment_params, model_params, loaders):
    
    # get params out
    num_rxns, tt_split, batch_size, recorded_batches, epochs, test_interval = experiment_params
    r_nec_ae, r_nec_opt, p_nec_ae, p_nec_opt = model_params
    train_loader, test_loader = loaders

    # log training
    experiment_log = Embedding_Exp_Log(num_rxns, tt_split, batch_size, epochs, recorded_batches)

    # training for n-1 epochs, don't save embeddings
    for epoch in range(1, epochs):
        train_loss, train_epoch_log, _ = train_tsi(r_nec_ae, p_nec_ae, r_nec_opt, p_nec_opt, train_loader)
        experiment_log.add_epoch(train_epoch_log)
        # return embs, then experiment_log.add_embs_and_batch(embs)
        # if epoch % 10 == 0:
        #     print(f"===== Training epoch {epoch:03d} complete with loss: {train_loss:.4f} ====")
    
    # final epoch and save embeddings
    train_loss, train_epoch_log, final_embs = train_tsi(r_nec_ae, p_nec_ae, r_nec_opt, p_nec_opt, train_loader)
    experiment_log.add_epoch(train_epoch_log)
    experiment_log.add_embs_and_batch(final_embs)
    epoch += 1
    print(f"===== Final training epoch {epoch:03d} complete with loss: {train_loss:.4f} ====")
    
    return experiment_log

def display_embeddings(exp_log):
    # rn just node embeddings

    final_embs_batched = exp_log.embeddings[-1]
    final_embs = [] # i.e. unbatched
    for (node_emb, ts_graph_emb, batch, r_graph_emb, p_graph_emb) in final_embs_batched:
        ts_node_emb_batch = to_dense_batch(node_emb, batch.x_ts_batch)[0] # just append tensors, not true/false values
        for mol_id, ts_node_emb in enumerate(ts_node_emb_batch):
            final_embs.append((ts_node_emb, ts_graph_emb[mol_id], r_graph_emb[mol_id], p_graph_emb[mol_id]))

    ts_graph_embs = [ts_graph_emb.detach().numpy() for (_, ts_graph_emb, _, _) in final_embs]
    r_graph_embs = [r_graph_emb.detach().numpy() for (_, _, r_graph_emb, _) in final_embs]
    p_graph_embs = [p_graph_emb.detach().numpy() for (_, _, _, p_graph_emb) in final_embs]

    cols = {'r': 'red', 'ts': 'orange', 'p': 'green'}

    fig, ax = plt.subplots(figsize = (8, 8))

    ax.scatter(*zip(*r_graph_embs), color = cols['r'])
    ax.scatter(*zip(*ts_graph_embs), color = cols['ts'])
    ax.scatter(*zip(*p_graph_embs), color = cols['p'])

    markers = [plt.Line2D([0,0], [0,0], color = color, marker = 'o', linestyle = '') for color in  cols.values()]
    ax.legend(markers, cols.keys())

    # title, axes
    num_rxns = exp_log.num_rxns
    train_ratio = exp_log.tt_split
    batch_size = exp_log.batch_size
    epochs = exp_log.get_num_epochs()
    title = f"{num_rxns} Reactions, Train Ratio: {train_ratio}, {epochs} Epochs, Batch Size: {batch_size}"
    ax.set_title(title)
    ax.set_ylabel('Graph Emb Dim 1')
    ax.set_xlabel('Graph Emb Dim 2')

    return fig, ax

    # create graph embs on train set: map node+edge embs -> graph emb
    #   - do for r_gt, p_gt, ts_gt separately; do for pre, post map ts_pred; display these five
    #   - if higher dim emb, use pca or tsne and see if difference
    # fig 4: compare test vs train embeddings
    #   - create test embs
    #   - plot cosine loss


def run_tsi_experiment(train_ratio = 0.8, batch_size = 5, epochs = 20, test_interval = 10):

    torch.set_printoptions(precision = 3, sci_mode = False)

    # data prep
    # print("Preparing data...")
    rxns = ReactionDataset(r'data')
    num_rxns = len(rxns)
    num_train = int(np.floor(train_ratio * num_rxns))
    to_follow = ['edge_index_r', 'edge_index_ts', 'edge_index_p', 'edge_attr_r', 'edge_attr_ts', 'edge_attr_p'
                'pos_r', 'pos_ts', 'pos_p', 'x_r', 'x_ts', 'x_p']
    train_loader = DataLoader(rxns[: num_train], batch_size = batch_size, follow_batch = to_follow)
    test_loader = DataLoader(rxns[num_train: ], batch_size = batch_size, follow_batch = to_follow)
    # print("Data prepared.\n")

    # model parameters
    # print("Preparing models...")
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
    # print("Models prepared.\n")

    # ts interpolation experiment: train model, get embeddings from train and test
    recorded_batches = []
    experiment_params = (num_rxns, train_ratio, batch_size, recorded_batches, epochs, test_interval)
    model_params = (r_nec_ae, r_nec_opt, p_nec_ae, p_nec_opt)
    loaders = (train_loader, test_loader)

    print("Starting TS interpolation experiment...")
    experiment_log = ts_interpolation(experiment_params, model_params, loaders)
    print("Completed experiment, use the experiment log to print results.")
    # print_embeddings(experiment_log)

    return experiment_log



if __name__ == "__main__":

    # torch.set_printoptions(precision = 3, sci_mode = False)
    exp_log = run_tsi_experiment()
    display_embeddings(exp_log)