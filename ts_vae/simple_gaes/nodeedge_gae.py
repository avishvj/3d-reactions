import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj

def unsorted_segment_sum(edge_attr, row, num_segments):
    result_shape = (num_segments, edge_attr.size(1))
    result = edge_attr.new_full(result_shape, 0) # init empty result tensor
    row = row.unsqueeze(-1).expand(-1, edge_attr.size(1))
    result.scatter_add_(0, row, edge_attr) # adds all values from tensor other int self at indices
    return result

# TODO:adj matrix needs to be of max num nodes length

class NodeEdge_AE(nn.Module):

    def __init__(self, in_node_nf = 11, in_edge_nf = 4, h_nf = 4, out_nf = 4, emb_nf = 2, act_fn = nn.ReLU(), device = 'cpu'):
        super(NodeEdge_AE, self).__init__()

        self.in_node_nf = in_node_nf
        self.in_edge_nf = in_edge_nf
        self.h_nf = h_nf
        self.out_nf = out_nf
        self.emb_nf = emb_nf
        self.device = device

        # encoder
        nel = NodeEdge_Layer(in_node_nf = in_node_nf, in_edge_nf = in_edge_nf, h_nf = h_nf, out_nf = out_nf,
                             act_fn = act_fn)
        self.add_module("NodeEdge", nel)
        # need final embedding for node and edge
        self.fc_node_emb = nn.Linear(out_nf, emb_nf)
        self.fc_edge_emb = nn.Linear(out_nf, emb_nf)

        self.to(device)
    
    def forward(self, node_feats, edge_index, edge_attr):
        # encode then decode
        node_emb, edge_emb = self.encode(node_feats, edge_index, edge_attr)
        recon_node_fs, recon_edge_fs, adj_pred = self.decode(node_emb, edge_emb)
        return node_emb, edge_emb, recon_node_fs, recon_edge_fs, adj_pred
    
    def encode(self, node_feats, edge_index, edge_attr):
        node_out, edge_out = self._modules["NodeEdge"](node_feats, edge_index, edge_attr)
        node_emb = self.fc_node_emb(node_out)
        edge_emb = self.fc_edge_emb(edge_out)
        return node_emb, edge_emb
    
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




class NodeEdge_Layer(nn.Module):

    def __init__(self, in_node_nf, in_edge_nf, h_nf, out_nf, bias = True, act_fn = nn.ReLU()):

        super(NodeEdge_Layer, self).__init__()

        # input dim is in_node_nf + unsorted_seg_sum output dim (which is == dim of edge_emb)
        self.node_mlp = nn.Sequential(nn.Linear(in_node_nf + out_nf, h_nf, bias = bias),
                                      act_fn,
                                      nn.Linear(h_nf, out_nf, bias = bias))
        
        # input dim is features for each bond i.e. node_fs of atoms either end + edge_fs
        self.edge_mlp = nn.Sequential(nn.Linear(in_node_nf * 2 + in_edge_nf, h_nf, bias = bias),
                                      act_fn,
                                      nn.Linear(h_nf, out_nf, bias = bias))
    
    def node_model(self, node_feats, edge_index, edge_attr):
        node_is, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, node_is, node_feats.size(0))
        node_in = torch.cat([node_feats, agg], dim = 1)
        return self.node_mlp(node_in)
    
    def edge_model(self, node_feats, edge_index, edge_attr):
        node_is, node_js = edge_index
        # get node feats for each bonded pair of atoms
        node_is_fs, node_js_fs = node_feats[node_is], node_feats[node_js]
        # node_is_fs, node_js_fs, edge_attr dim: [n_bonded, 11], [n_bonded, 11], [n_bonded, 4]
        edge_in = torch.cat([node_is_fs, node_js_fs, edge_attr], dim = 1)
        return self.edge_mlp(edge_in)
    
    def forward(self, node_feats, edge_index, edge_attr):
        edge_out = self.edge_model(node_feats, edge_index, edge_attr)
        node_out = self.node_model(node_feats, edge_index, edge_out)
        return node_out, edge_out


class Node_AE(nn.Module):
    # node_ae with node_layer node now, then layers, then act fn, then train
    # node+edge_layer
    
    def __init__(self, in_node_nf = 11, in_edge_nf = 4, h_nf = 4, out_nf = 4, emb_nf = 2, act_fn = nn.ReLU(), device = 'cpu'):
        super(Node_AE, self).__init__()

        self.in_node_nf = in_node_nf
        self.in_edge_nf = in_edge_nf
        self.h_nf = h_nf
        self.out_nf = out_nf
        self.emb_nf = emb_nf
        self.device = device

        # encoder
        nl = Node_Layer(in_nf = in_node_nf + in_edge_nf, h_nf = h_nf, out_nf = out_nf, act_fn = act_fn) # , edges_nf = 4) # , act_fn = act_fn)
        self.add_module("Node", nl)
        self.fc_emb = nn.Linear(out_nf, emb_nf) 

        self.to(device)
    
    def forward(self, node_feats, edge_index, edge_attr):
        # encode then decode
        node_emb = self.encode(node_feats, edge_index, edge_attr)
        recon_node_fs, adj_pred = self.decode(node_emb)
        return node_emb, recon_node_fs, adj_pred

    def encode(self, node_feats, edge_index, edge_attr):
        # node layer then linear
        node_feats = self._modules["Node"](node_feats, edge_index, edge_attr)
        return self.fc_emb(node_feats)

    def decode(self, node_emb):
        # decode to node features and adj matrix
        # TODO: add decode_to_adj params here?
        recon_node_fs = self.decode_to_node_fs(node_emb)
        adj_pred = self.decode_to_adj(node_emb)
        return recon_node_fs, adj_pred

    def decode_to_node_fs(self, node_emb):
        # simple linear layer from embedding to node features
        emb_to_fs = nn.Linear(self.emb_nf, self.in_node_nf)
        recon_node_fs = emb_to_fs(node_emb)
        return recon_node_fs

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



class Node_Layer(nn.Module):

    def __init__(self, in_nf, h_nf, out_nf, bias = True, act_fn = nn.ReLU()):
        super(Node_Layer, self).__init__()
        # first nn.Linear(in, out): in :- num_node_fs + num_agg_fs == num_node_fs + num_edge_aggr_fs
        self.node_mlp = nn.Sequential(nn.Linear(in_nf, h_nf, bias = bias),
                                      act_fn,
                                      nn.Linear(h_nf, out_nf, bias = bias))
    
    def node_model(self, node_feats, edge_index, edge_attr):
        node_is, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, node_is, node_feats.size(0))
        node_in = torch.cat([node_feats, agg], dim = 1)
        return self.node_mlp(node_in)
    
    def forward(self, node_feats, edge_index, edge_attr):
        node_feats = self.node_model(node_feats, edge_index, edge_attr)
        return node_feats
    

### train and test functions

def train_ne_ae(ne_ae, opt, loader):
    res = {'total_loss': 0, 'counter': 0, 'total_loss_arr': [], 
           'node_recon_loss_arr': [], 'edge_recon_loss_arr': [], 'adj_loss_arr': []}
    
    for i, rxn_batch in enumerate(loader):

        ne_ae.train()
        opt.zero_grad()

        # gen node emb, edge emb, adj
        node_feats, edge_index, edge_attr = rxn_batch.x, rxn_batch.edge_index, rxn_batch.edge_attr
        batch_size, batch_vec = len(rxn_batch.idx), rxn_batch.batch
        node_emb, edge_emb, recon_node_fs, recon_edge_fs, adj_pred = ne_ae(node_feats, edge_index, edge_attr)

        # ground truth values
        adj_gt = to_dense_adj(edge_index).squeeze(dim = 0)
        assert adj_gt.shape == adj_pred.shape, f"Your adjacency matrices don't have the same shape! \
                GT shape: {adj_gt.shape}, Pred shape: {adj_pred.shape}, Batch size: {batch_size}, \
                   Node fs shape: {node_feats.shape} "

        # losses and opt step
        node_recon_loss = F.mse_loss(recon_node_fs, node_feats)
        edge_recon_loss = F.mse_loss(recon_edge_fs, edge_attr)
        adj_loss = F.binary_cross_entropy(adj_pred, adj_gt)
        total_loss = node_recon_loss + edge_recon_loss + adj_loss
        total_loss.backward()
        opt.step()

        # record batch results
        res['total_loss'] += total_loss.item() * batch_size
        res['counter'] += batch_size
        res['node_recon_loss_arr'].append(node_recon_loss.item())
        res['edge_recon_loss_arr'].append(edge_recon_loss.item())
        res['adj_loss_arr'].append(adj_loss.item())
        res['total_loss_arr'].append(total_loss.item())
    
    return res['total_loss'] / res['counter'], res



        

def train_node_ae(node_ae, opt, loader):
    # standard loader

    # use dict to record results, TODO: experiment dataclass: loss, epoch, batch_size
    res = {'loss': 0, 'counter': 0, 'loss_arr': [], 'adj_loss_arr': [], 'node_recon_loss_arr': []}

    for i, rxn_batch in enumerate(loader):

        node_ae.train()
        opt.zero_grad()

        # generate node embeddings and predicted adj matrix
        node_feats, edge_index, edge_attr = rxn_batch.x, rxn_batch.edge_index, rxn_batch.edge_attr
        batch_size, batch_vec = len(rxn_batch.idx), rxn_batch.batch
        node_emb, recon_node_fs, adj_pred = node_ae(node_feats, edge_index, edge_attr)

        # ground truth adj matrix; if add batch vec, you get two, if add edge_attr get x4 in dim
        adj_gt = to_dense_adj(edge_index = edge_index).squeeze(dim = 0)
        assert adj_gt.shape == adj_pred.shape, "Your adjacency matrices don't have the same shape!"

        # node recon loss
        node_recon_loss = F.mse_loss(recon_node_fs, node_feats)

        # adj loss
        adj_loss = F.binary_cross_entropy(adj_pred, adj_gt)

        # simple combination of loss right now
        loss = node_recon_loss + adj_loss
        loss.backward() 
        opt.step()

        # record batch results
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['adj_loss_arr'].append(adj_loss.item())
        res['node_recon_loss_arr'].append(node_recon_loss.item())
        res['loss_arr'].append(loss.item())
    
    return res['loss'] / res['counter'], res

# for other adj loss?        
# adj_gt, adj_pred = adj_gt.detach().cpu().numpy(), adj_pred.detach().cpu().numpy()
# loss = roc_auc_score(adj_gt, adj_pred) # , average_precision_score(adj_gt, adj_pred), other metrics

# could just add train/test tags in original func as mostly same funcs
# TODO: remove opt
def test_node_ae(node_ae, opt, loader):
    
    res = {'loss': 0, 'counter': 0, 'loss_arr': [], 'adj_loss_arr': [], 'node_recon_loss_arr': []}

    for i, rxn_batch in enumerate(loader):

        # test mode
        node_ae.eval()

        # generate node embeddings and predicted adj matrix
        node_feats, edge_index, edge_attr = rxn_batch.x, rxn_batch.edge_index, rxn_batch.edge_attr
        batch_size, batch_vec = len(rxn_batch.idx), rxn_batch.batch
        node_emb, recon_node_fs, adj_pred = node_ae(node_feats, edge_index, edge_attr)

        # ground truth adj matrix; if add batch vec, you get two, if add edge_attr get x4 in dim
        adj_gt = to_dense_adj(edge_index = edge_index).squeeze(dim = 0)
        assert adj_gt.shape == adj_pred.shape, "Your adjacency matrices don't have the same shape!"
        
        # node recon loss
        node_recon_loss = F.mse_loss(recon_node_fs, node_feats)

        # adj loss
        adj_loss = F.binary_cross_entropy(adj_pred, adj_gt)

        # simple combination of loss right now
        loss = node_recon_loss + adj_loss

        # record batch results
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['adj_loss_arr'].append(adj_loss.item())
        res['node_recon_loss_arr'].append(node_recon_loss.item())
        res['loss_arr'].append(loss.item())
    
    return res['loss'] / res['counter'], res


