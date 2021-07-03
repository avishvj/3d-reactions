import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from experiments.exp_utils import BatchLog, EpochLog
from ..utils import reset
import operator

def average(r, p):
    if r.shape != p.shape:
        # this happens on edge features since bonds broken and formed
        # current hacky soln: just average min features, TODO: proper soln
        if r.size(0) != p.size(0):
            min_num_fs = min(r.size(0), p.size(0))
            return (r[0:min_num_fs, ] + p[0:min_num_fs, ]) / 2
        elif r.size(1) != p.size(1):
            min_num_fs = min(r.size(1), p.size(1))
            return (r[:, 0:min_num_fs] + p[:, 0:min_num_fs]) / 2
    else:
        return (r + p) / 2

combination_funcs = {'average': average}

class TS_AE(nn.Module):
    """ A graph autoencoder with reactant-product (i.e. two graph) combination logic.
    Encodes to r/p/ts_gt mappings, combines to get ts_premapped/ts_postmapped, decodes to r/p/ts_gt/ts_premap/ts_postmap. 
    Use the three TS mappings/decodings for relative comparison.
    Sources of randomness/variability:
        - NEC layer: coord_edge_mlp: xav_uniform init
        - Training: SGD i.e. batches, other stuff?    
    """

    def __init__(self, encoder, decoder, comb_func = 'average', device = 'cpu'):
        super(TS_AE, self).__init__()

        # architecture
        self.encoder = encoder
        self.decoder = decoder

        # R and P combination function
        assert comb_func in combination_funcs.keys(), f"Combination function not valid. Please pick from {combination_funcs.keys()}"
        # self.comb_func = combination_funcs[self.comb_func]
        self.comb_func = comb_func

        self.to(device)
    
    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)
        
    def forward(self, r_params, p_params, ts_params, batch_node_vecs, premap = True, postmap = True, coord_baseline = False):
        # params = node_feats, edge_index, edge_attr, coords
        # mapped = node_emb, edge_emb, graph_emb, coord_out
        # decoded = recon_node_fs, recon_edge_fs, adj_pred

        # ground truth mols, encoded and decoded
        r_mapped, p_mapped, ts_gt_mapped = self.encode_mols_gt(r_params, p_params, ts_params, batch_node_vecs)
        r_decoded, p_decoded, ts_gt_decoded = self.decode_mols_gt(r_mapped, p_mapped, ts_gt_mapped)

        # record autoencoder pass
        batch_ae_res = {}
        batch_ae_res['r'] = (r_mapped, r_decoded)
        batch_ae_res['p'] = (p_mapped, p_decoded)
        batch_ae_res['ts_gt'] = (ts_gt_mapped, ts_gt_decoded)
        batch_ae_res['batch_node_vecs'] = batch_node_vecs

        if premap:
            ts_premapped = self.premap(r_params, p_params, batch_node_vecs)
            ts_premap_decoded = self.decoder(*ts_premapped)
            batch_ae_res['ts_premap'] = (ts_premapped, ts_premap_decoded)
        
        if postmap:
            ts_postmapped = self.postmap(r_mapped, p_mapped)
            ts_postmap_decoded = self.decoder(*ts_postmapped)
            batch_ae_res['ts_postmap'] = (ts_postmapped, ts_postmap_decoded)
        
        if coord_baseline:
            ts_coord_baseline = self.coord_baseline(r_params, p_params)
            batch_ae_res['coord_baseline'] = ts_coord_baseline

        return batch_ae_res
    
    ### main functions used in forward

    def combine_r_and_p(self, r_feats, p_feats, comb_func = 'average'):
        assert len(r_feats) == len(p_feats), f"Should map reactant (len: {len(r_feats)}) and \
               product (len: {len(p_feats)}) to same number of features."
        comb_func = combination_funcs[comb_func]
        ts_zip = list(zip(r_feats, p_feats))
        ts_mapped_comb = [comb_func(r_feat, p_feat) for (r_feat, p_feat) in ts_zip]
        return ts_mapped_comb

    def encode_mols_gt(self, r_params, p_params, ts_params, batch_node_vecs):
        r_batch_vec, p_batch_vec, ts_batch_vec = batch_node_vecs
        r_mapped = self.encoder(*r_params, r_batch_vec)
        p_mapped = self.encoder(*p_params, p_batch_vec)
        ts_gt_mapped = self.encoder(*ts_params, ts_batch_vec)
        return r_mapped, p_mapped, ts_gt_mapped
    
    def decode_mols_gt(self, r_mapped, p_mapped, ts_gt_mapped):
        r_decoded = self.decoder(*r_mapped)
        p_decoded = self.decoder(*p_mapped)
        ts_gt_decoded = self.decoder(*ts_gt_mapped)
        return r_decoded, p_decoded, ts_gt_decoded
    
    ### baselines

    def coord_baseline(self, r_params, p_params):
        _, _, _, r_coords = r_params
        _, _, _, p_coords = p_params
        return self.combine_r_and_p(r_coords, p_coords, 'average')

    def premap(self, r_params, p_params, batch_node_vecs):
        # average r/p raw features as ts params, then encode to ts emb, TODO: r+p/2 batch vec
        # NOTE: should fail if r has more edges than p, will have to use a min max in average()
        r_batch_vec, p_batch_vec, _ = batch_node_vecs
        ts_params_premap = self.combine_r_and_p(r_params, p_params, 'average')
        ts_params_premap[1] = ts_params_premap[1].long()
        return self.encoder(*ts_params_premap, r_batch_vec)

    def postmap(self, r_mapped, p_mapped):
        # take mapped r/p raw feats (i.e. r/p embs), and average
        return self.combine_r_and_p(r_mapped, p_mapped, 'average')
    
    ### loss functions

    def recon_loss(self, r_params, p_params, ts_params, max_num_atoms, batch_ae_res, train_on_ts = False):
        # params = node_feats, edge_index, edge_attr, coords
        # mapped = node_emb, edge_emb, graph_emb, coord_out
        # decoded = recon_node_fs, recon_edge_fs, adj_pred
        # losses = node_loss, adj_loss, coord_loss, batch_loss

        r_losses = self.ind_recon_loss(r_params, max_num_atoms, batch_ae_res, 'r')
        p_losses = self.ind_recon_loss(p_params, max_num_atoms, batch_ae_res, 'p')
        
        combined_losses = tuple(map(operator.add, r_losses, p_losses))

        if train_on_ts:
            ts_losses = self.ind_recon_loss(ts_params, max_num_atoms, batch_ae_res, 'ts_gt')
            combined_losses = tuple(map(operator.add, combined_losses, ts_losses))
        
        return combined_losses

    def ind_recon_loss(self, gt_params, max_num_atoms, batch_ae_res, key):
        # for individual mols e.g. r_params vs batch_ae_res[key = 'r']

        gt_node_feats, gt_edge_index, gt_edge_attr, gt_coords = gt_params
        mapped, decoded = batch_ae_res[key]
        node_emb, edge_emb, graph_emb, coord_out = mapped
        recon_node_fs, recon_edge_fs, adj_pred = decoded

        # adjacency matrix
        adj_gt = to_dense_adj(gt_edge_index, max_num_nodes = max_num_atoms).squeeze(dim = 0)
        assert adj_gt.shape == adj_pred.shape, f"Your adjacency matrices don't have the same shape!" 

        # losses
        node_loss = F.mse_loss(recon_node_fs, gt_node_feats) # scale: e-1 --> e-2, 10 epochs: 0.7 -> 0.05
        adj_loss = F.binary_cross_entropy(adj_pred, adj_gt) # scale: e-1, 10 epochs: 0.5 -> 0.4
        coord_loss = torch.sqrt(F.mse_loss(coord_out, gt_coords)) # scale: e-1, 10 epochs: 1.1 -> 0.4
        batch_loss = node_loss + adj_loss + coord_loss
        
        return node_loss, adj_loss, coord_loss, batch_loss

### training and testing

def train(ts_ae, ts_opt, loader, test = False, train_on_ts = False):

    epoch_log = EpochLog()
    epoch_ae_res = []
    total_loss = 0

    for batch_id, rxn_batch in enumerate(loader):

        # train mode + zero gradients
        if not test:
            ts_ae.train()
            ts_opt.zero_grad()
        else:
            ts_ae.eval()
        
        # init r, p, ts params
        r_params = rxn_batch.x_r, rxn_batch.edge_index_r, rxn_batch.edge_attr_r, rxn_batch.pos_r
        p_params = rxn_batch.x_p, rxn_batch.edge_index_p, rxn_batch.edge_attr_p, rxn_batch.pos_p
        ts_params = rxn_batch.x_ts, rxn_batch.edge_index_ts, rxn_batch.edge_attr_ts, rxn_batch.pos_ts
        batch_size = len(rxn_batch.idx)
        max_num_atoms = sum(rxn_batch.num_atoms).item() # add this in because sometimes we get hanging atoms if bonds broken
        batch_node_vecs = rxn_batch.x_r_batch, rxn_batch.x_p_batch, rxn_batch.x_ts_batch # for recreating graphs

        # run batch pass of ae with params
        batch_ae_res = ts_ae(r_params, p_params, ts_params, batch_node_vecs)

        # loss and step
        node_loss, adj_loss, coord_loss, batch_loss = ts_ae.recon_loss(r_params, p_params, ts_params, max_num_atoms, batch_ae_res, train_on_ts)
        total_loss += batch_loss

        if not test:
            batch_loss.backward()
            ts_opt.step()
        
        # log batch results
        batch_log = BatchLog(batch_size, batch_id, max_num_atoms, batch_node_vecs,
                             coord_loss.item(), adj_loss.item(), node_loss.item(), batch_loss.item())
        epoch_log.add_batch(batch_log)
        epoch_ae_res.append(batch_ae_res)
    
    return total_loss / batch_id, epoch_log, epoch_ae_res


    
    




    

