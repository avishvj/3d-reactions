import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

# basically lifted from original egnn: https://github.com/vgsatorras/egnn/

def unsorted_segment_sum(edge_attr, row, num_segments):
    result_shape = (num_segments, edge_attr.size(1))
    result = edge_attr.new_full(result_shape, 0) # init empty result tensor
    row = row.unsqueeze(-1).expand(-1, edge_attr.size(1))
    result.scatter_add_(0, row, edge_attr) # adds all values from tensor other int self at indices
    return result


class NodeEdgeCoord_AE(nn.Module):

    def __init__(self, in_node_nf = 11, in_edge_nf = 4, h_nf = 4, out_nf = 4, emb_nf = 2, act_fn = nn.ReLU(), device = 'cpu'):
        super(NodeEdgeCoord_AE, self).__init__()
        
        # standard params
        self.in_node_nf = in_node_nf
        self.in_edge_nf = in_edge_nf
        self.h_nf = h_nf
        self.out_nf = out_nf
        self.emb_nf = emb_nf
        self.device = device

        # coord params?

        # encoder
        necl = NodeEdgeCoord_Layer(in_node_nf = in_node_nf, in_edge_nf = in_edge_nf,
                                   h_nf = h_nf, out_nf = out_nf, act_fn = act_fn)
        self.add_module("NodeEdgeCoord", necl)
        # final emb for node and edge
        self.fc_node_emb = nn.Linear(out_nf, emb_nf)
        out_edge_nf = in_edge_nf
        self.fc_edge_emb = nn.Linear(out_edge_nf, emb_nf)

        self.to(device)
    
    def forward(self, node_feats, edge_index, edge_attr, coords):
        # encode then decode
        node_emb, edge_emb, coord_out = self.encode(node_feats, edge_index, edge_attr, coords)
        recon_node_fs, recon_edge_fs, adj_pred = self.decode(node_emb, edge_emb)
        return node_emb, edge_emb, recon_node_fs, recon_edge_fs, adj_pred, coord_out

    def encode(self, node_feats, edge_index, edge_attr, coords):
        node_out, edge_out, coord_out = self._modules["NodeEdgeCoord"](node_feats, edge_index, edge_attr, coords)
        node_emb = self.fc_node_emb(node_out)
        edge_emb = self.fc_edge_emb(edge_out)
        return node_emb, edge_emb, coord_out

    ### TODO: these decode funcs are exactly the same as NodeEdge - maybe worth creating base class of both
    
    def decode(self, node_emb, edge_emb):
        # decode to edge_feats, node_feats, adj
        recon_node_fs = self.decode_to_node_fs(node_emb)
        recon_edge_fs = self.decode_to_edge_fs(edge_emb)
        adj_pred = self.decode_to_adj(node_emb)
        return recon_node_fs, recon_edge_fs, adj_pred

    def decode_to_node_fs(self, node_emb):
        # simple linear layer from node embedding to node features
        emb_to_fs = nn.Linear(self.emb_nf, self.in_node_nf)
        recon_node_fs = emb_to_fs(node_emb)
        return recon_node_fs

    def decode_to_edge_fs(self, edge_emb):
        # simple linear layer from edge embedding to edge features
        emb_to_fs = nn.Linear(self.emb_nf, self.in_edge_nf)
        recon_edge_fs = emb_to_fs(edge_emb)
        return recon_edge_fs

    def decode_to_adj(self, x, W = 3, b = -1, linear_sig = True, remove_diag = True):
                # num_nodes dim: [num_nodes, 2]
        # use number of nodes in batch as adj matrix dimensions (obviously!)
        # generate differences between node embeddings as adj matrix
        # W, b: weights and biases for linear layer. push nodes thru linear layer at end
        # TODO: rn, decode to adj matrix. also want to decode to original node_feats
        # TODO: is this basically just: torch.sigmoid(torch.matmul(x, x.t()))?
        # need to decode to adj and nodes

        x_a = x.unsqueeze(0) # dim: [1, num_nodes, 2]
        x_b = torch.transpose(x_a, 0, 1) # dim: [num_nodes, 1, 2], t.t([_, dim to t, dim to t])

        X = (x_a - x_b) ** 2  # dim: [num_nodes, num_nodes, 2]
        
        num_nodes = x.size(0) # num_nodes (usually as number of nodes in batch)
        X = X.view(num_nodes ** 2, -1) # dim: [num_nodes^2, 2] to apply sum 

        # (lin_sig or not) layer, dim=1 sums to dim=[num_nodes^2]
        # gives porbabilistic adj matrix
        X = torch.sigmoid(W * torch.sum(X, dim = 1) + b) if linear_sig else torch.sum(X, dim = 1)

        adj_pred = X.view(num_nodes, num_nodes) # dim: [num_nodes, num_nodes]
        if remove_diag: # TODO: the pyg method adds self-loops, what do I want?
            adj_pred = adj_pred * (1 - torch.eye(num_nodes).to(self.device))
        
        return adj_pred


class NodeEdgeCoord_Layer(nn.Module):
    
    def __init__(self, in_node_nf, in_edge_nf, h_nf, out_nf, bias = True, act_fn = nn.ReLU()):

        super(NodeEdgeCoord_Layer, self).__init__()

        # node_norm : LayerNorm; coord_norm: define (follow se3); edge_norm: required?

        # setting feature and mlp dimensions
        out_edge_nf = in_edge_nf
        coord_dim = 3
        radial_dim = 1
        h_coord_nf = radial_dim * 2 # arbitrary, just between num_edge_fs and 1

        # mlp input dims
        in_node_mlp_nf = in_node_nf + out_edge_nf + coord_dim # node_feats + agg + coords
        in_edge_mlp_nf = (in_node_nf * 2) + in_edge_nf + radial_dim
        in_coord_mlp_nf = in_edge_nf # just the number of edge features

        # node mlp
        self.node_mlp = nn.Sequential(nn.Linear(in_node_mlp_nf, h_nf, bias = bias),
                                      act_fn,
                                      nn.Linear(h_nf, out_nf, bias = bias))

        # edge mlp
        self.edge_mlp = nn.Sequential(nn.Linear(in_edge_mlp_nf, h_nf, bias = bias),
                                      act_fn,
                                      nn.Linear(h_nf, out_edge_nf, bias = bias))
        
        # coord mlp: no bias, final layer has xavier_uniform init [following orig]
        layer = nn.Linear(h_coord_nf, radial_dim, bias = False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(nn.Linear(in_coord_mlp_nf, h_coord_nf), nn.ReLU(), layer)
    
    def node_model(self, node_feats, edge_index, edge_attr, coords):
        # NOTE: edge_attr here is actually edge_mlp(original edge_attr)
        # NOTE: currently using coords as feature... maybe bad idea. doesn't in original.
        node_is, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, node_is, node_feats.size(0))
        node_in = torch.cat([node_feats, agg, coords], dim = 1)
        return self.node_mlp(node_in)
    
    def edge_model(self, node_feats, edge_index, edge_attr, radial):
        node_is, node_js = edge_index
        # get node feats for each bonded pair of atoms
        node_is_fs, node_js_fs = node_feats[node_is], node_feats[node_js]
        edge_in = torch.cat([node_is_fs, node_js_fs, edge_attr, radial], dim = 1)
        return self.edge_mlp(edge_in)
    
    def coord_model(self, edge_index, edge_attr, coords, coord_diffs):
        # radially normed coord differences (i.e. bond lengths) * coord_mlp(edges)
        # == normed bond lengths * edge_out
        # NOTE: edge_attr here is actually edge_mlp(original edge_attr)
        atom_is, _ = edge_index
        trans = coord_diffs * self.coord_mlp(edge_attr)
        agg = unsorted_segment_sum(trans, atom_is, coords.size(0))
        coords += agg
        return coords

    def coord_to_radial(self, edge_index, coords):
        # calc coordinate differences between bonded atoms (i.e. bond lengths)
        atom_is, atom_js = edge_index
        coord_diffs = coords[atom_is] - coords[atom_js]
        # normalise coords, TODO: alternative coord_norm as func or class
        radial = torch.sum((coord_diffs)**2, 1).unsqueeze(1)
        norm = torch.sqrt(radial) + 1
        normed_coord_diffs = coord_diffs / norm
        return radial, normed_coord_diffs
    
    def forward(self, node_feats, edge_index, edge_attr, coords):
        # maybe shouldn't be using coords in node_out
        # ideally would want to use node and edge features for final coords?
        radial, coord_diffs = self.coord_to_radial(edge_index, coords)
        edge_out = self.edge_model(node_feats, edge_index, edge_attr, radial)
        coord_out = self.coord_model(edge_index, edge_out, coords, coord_diffs)
        node_out = self.node_model(node_feats, edge_index, edge_out, coord_out)
        return node_out, edge_out, coord_out


### train and test functions

def train_nec_ae(nec_ae, opt, loader):

    res = {'total_loss': 0, 'counter': 0, 'total_loss_arr': [], 'coord_loss_arr': [],
           'node_recon_loss_arr': [], 'edge_recon_loss_arr': [], 'adj_loss_arr': []}
    
    # not sure how coords update would work for full dataset?
    # coord_updates = {'batch_counter': 0, 'batch_coords': []}

    for i, rxn_batch in enumerate(loader):

        nec_ae.train()
        opt.zero_grad()

        # gen node emb, edge emb, adj, coords
        node_feats, edge_index, edge_attr, coords = rxn_batch.x, rxn_batch.edge_index, rxn_batch.edge_attr, rxn_batch.pos
        batch_size, batch_vec = len(rxn_batch.idx), rxn_batch.batch
        node_emb, edge_emb, recon_node_fs, recon_edge_fs, adj_pred, coord_out = nec_ae(node_feats, edge_index, edge_attr, coords)

        # ground truth values
        adj_gt = to_dense_adj(edge_index).squeeze(dim = 0)
        assert adj_gt.shape == adj_pred.shape, f"Your adjacency matrices don't have the same shape! \
                GT shape: {adj_gt.shape}, Pred shape: {adj_pred.shape}, Batch size: {batch_size}, \
                   Node fs shape: {node_feats.shape} "

        # losses and opt step
        node_recon_loss = F.mse_loss(recon_node_fs, node_feats)
        edge_recon_loss = F.mse_loss(recon_edge_fs, edge_attr)
        adj_loss = F.binary_cross_entropy(adj_pred, adj_gt)
        coord_loss = F.mse_loss(coord_out, coords)
        total_loss = node_recon_loss + edge_recon_loss + adj_loss + coord_loss
        total_loss.backward()
        opt.step()

        # record batch results
        res['total_loss'] += total_loss.item() * batch_size
        res['counter'] += batch_size
        res['node_recon_loss_arr'].append(node_recon_loss.item())
        res['edge_recon_loss_arr'].append(edge_recon_loss.item())
        res['adj_loss_arr'].append(adj_loss.item())
        res['coord_loss_arr'].append(coord_loss.item())
        res['total_loss_arr'].append(total_loss.item())
    
    return res['total_loss'] / res['counter'], res

def test_nec_ae(nec_ae, loader):

    res = {'total_loss': 0, 'counter': 0, 'total_loss_arr': [], 'coord_loss_arr': [],
           'node_recon_loss_arr': [], 'edge_recon_loss_arr': [], 'adj_loss_arr': []}
    
    # not sure how coords update would work for full dataset?
    # coord_updates = {'batch_counter': 0, 'batch_coords': []}

    for i, rxn_batch in enumerate(loader):

        nec_ae.eval()

        # gen node emb, edge emb, adj, coords
        node_feats, edge_index, edge_attr, coords = rxn_batch.x, rxn_batch.edge_index, rxn_batch.edge_attr, rxn_batch.pos
        batch_size, batch_vec = len(rxn_batch.idx), rxn_batch.batch
        node_emb, edge_emb, recon_node_fs, recon_edge_fs, adj_pred, coord_out = nec_ae(node_feats, edge_index, edge_attr, coords)

        # ground truth values
        adj_gt = to_dense_adj(edge_index).squeeze(dim = 0)
        assert adj_gt.shape == adj_pred.shape, f"Your adjacency matrices don't have the same shape! \
                GT shape: {adj_gt.shape}, Pred shape: {adj_pred.shape}, Batch size: {batch_size}, \
                   Node fs shape: {node_feats.shape} "

        # losses and opt step
        node_recon_loss = F.mse_loss(recon_node_fs, node_feats)
        edge_recon_loss = F.mse_loss(recon_edge_fs, edge_attr)
        adj_loss = F.binary_cross_entropy(adj_pred, adj_gt)
        coord_loss = F.mse_loss(coord_out, coords)
        total_loss = node_recon_loss + edge_recon_loss + adj_loss + coord_loss

        # record batch results
        res['total_loss'] += total_loss.item() * batch_size
        res['counter'] += batch_size
        res['node_recon_loss_arr'].append(node_recon_loss.item())
        res['edge_recon_loss_arr'].append(edge_recon_loss.item())
        res['adj_loss_arr'].append(adj_loss.item())
        res['coord_loss_arr'].append(coord_loss.item())
        res['total_loss_arr'].append(total_loss.item())
    
    return res['total_loss'] / res['counter'], res

def main(epochs = 20, test_interval = 5):
    final_res = {'epochs': [], 'train_loss_arr': [], 'train_res_arr': [], 
                'test_loss_arr': [], 'test_res_arr': [], 'best_test': 1e10, 'best_epoch': 0}

    # nec_gae.reset_parameters()

    for epoch in range(1, epochs + 1):
        
        train_loss, train_res = train_nec_ae(nec_ae, nec_opt, train_loaders['r'])
        final_res['train_loss_arr'].append(train_loss)
        final_res['train_res_arr'].append(train_res)
        print(f"===== Training epoch {epoch:03d} complete with loss: {train_loss:.4f} ====")
        
        if epoch % test_interval == 0:
        
            test_loss, test_res = test_nec_ae(nec_ae, test_loaders['r'])
            final_res['test_loss_arr'].append(test_loss)
            final_res['test_res_arr'].append(test_res)
            print(f'===== Testing epoch: {epoch:03d}, Loss: {test_loss:.4f} ===== \n')
            
            if test_loss < final_res['best_test']:
                final_res['best_test'] = test_loss
                final_res['best_epoch'] = epoch
    
    return final_res

