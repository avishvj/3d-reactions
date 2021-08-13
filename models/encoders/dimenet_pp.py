import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Embedding
from torch.serialization import register_package

from torch_geometric.nn import radius_graph
from torch_geometric.nn.acts import swish, ReLU
from torch_scatter import scatter
from torch_sparse import SparseTensor

from ts_ae.ae import TS_AE


# pytorch version of dimenet++, adapted from dig and source
# https://github.com/klicperajo/dimenet/blob/master/dimenet/model/dimenet_pp.py

# NOTE: will have to redo with separate updates for node, edge, graph like 3D MP

class DimeNetPP(nn.Module):

    def __init__(self, cutoff_val, envelope_exp, n_spherical, n_radial, device = 'cpu'):
        super(DimeNetPP, self).__init__()

        self.cutoff_val = cutoff_val

        # edge init
        # TODO: they have a special initial init for edges but then standard updates for node and graph init


        # cosine basis function expansion
        self.emb = DGEmbedding(cutoff_val, envelope_exp, n_spherical, n_radial)

        self.device = device
        self.to(device)

    
    def forward(self, atomic_ns, coords, batch_node_vec):

        # processing sequence
        #   dist -> rbf
        #   dist+angle -> sbf
        #   
        #   rbf -> emb
        #   rbf + sbf -> interaction
        #   
        #   sum(emb, interactions) -> output mlp

        # functions needed
        #   interatomic dist
        #   neighbour angles from triplets

        # dist to rbf
        edge_index = radius_graph(coords, self.cutoff_val, batch_node_vec)
        num_nodes = atomic_ns.size(0)

        dists, angles, node_is, node_js, kj, ji = xyz_to_dg(coords, edge_index, num_nodes)

        dist_embs, angle_embs = self.emb(dists, angles, kj)

        # init edge, node, graph feats
        #edge_attr = self.

        return
    
    def calc_neighbour_angles(self, R, i, j, k):
        
        # R_i = 

        return 
    

class FeatsInitLayer(nn.Module):

    def __init__(self, n_radial, h_nf, edge_act = swish):
        super(FeatsInitLayer, self).__init__()


        # edge init
        self.edge_act = edge_act
        self.emb = Embedding(95, h_nf)
        self.lin_rbf0 = Linear(n_radial, h_nf)
        self.lin = Linear(3 * h_nf, h_nf)
        self.lin_rbf1 = Linear(n_radial, h_nf, bias = False)

        # node init
        self.node_act = edge_act # TODO: change later




        pass

    def forward(self, coords, embs, node_is, node_js):

        rbf, _ = embs
        es = self.edge_model(coords, rbf, node_is, node_js)


        pass
    
    def edge_model(self, coords, rbf, node_is, node_js):
        x = self.emb(coords)
        rbf0 = self.act(self.lin_rbf0(rbf))
        e1 = self.act(self.lin(torch.cat([x[node_is], x[node_js], rbf0], dim=-1)))
        e2 = self.lin_rbf1(rbf) * e1
        return e1, e2
    
    def node_model(self, es, node_is):
        _, e2 = es
        node_feats = scatter(e2, node_is, dim = 0)
        node_feats = self.lin_up(node_feats)
        pass



class DGEmbedding(nn.Module):
    # distance geometry embedding
    def __init__(self, cutoff_val, envelope_exp, n_spherical, n_radial):
        super(DGEmbedding, self).__init__()
        
        # both embs
        self.cutoff_val = cutoff_val
        self.envelope_exp = envelope_exp
        self.n_radial = n_radial

        # dist emb
        self.freq = torch.nn.Parameter(torch.Tensor(n_radial))

        # angle emb
        self.n_spherical = n_spherical
        # TODO: there are a lot of things here... might be worth importing DIG
    
    def forward(self, dists, angles, kj):
        dist_embs = self.create_dist_embs(dists)
        angle_embs = self.create_angle_embs(dists, angles, kj)
        return dist_embs, angle_embs
    
    def create_dist_embs(self, dists):
        dists = dists.unsqueeze(-1) / self.cutoff
        return self.envelope(dists) * (self.freq * dists).sin()
    
    def create_angle_embs(self, dists, angles, kj):
        dists = dists / self.cutoff_val
        rbf = torch.stack([f(dists) for f in self.bessel_funcs], dim = 1)
        rbf = self.envelope(dists).unsqueeze(-1) * rbf
        cbf = torch.stack([f(angles) for f in self.sph_funcs], dim = 1)
        n, k = self.n_spherical, self.n_radial
        out = (rbf[kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out
    
    def envelope(self, x):
        p = self.envelope_exp + 1
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = p * (p + 1) / 2
        p0 = x.pow(p-1)
        p1 = p0 * x
        p2 = p1 * x
        return 1. / x + a*p0 + b*p1 + c*p2
    

def xyz_to_dg(coords, edge_index, num_nodes, device):
    # return distance geometry params: distances, angles
    # repeat_interleave: repeats elements of a tensor

    node_js, node_is = edge_index # j->i here so k->j->i later

    # calculate distances
    dists = (coords[node_is] - coords[node_js]).pow(2).sum(dim = -1).sqrt()
    
    # init number of edges for each triplet
    value = torch.arange(node_js.size(0), device)
    adj_t = SparseTensor(node_is, node_js, value, (num_nodes, num_nodes))
    adj_t_row = adj_t[node_js]
    num_triplets = adj_t_row.set_value(None).sum(dim = 1).to(torch.long)

    # node indices (k->j->i)
    i = node_is.repeat_interleave(num_triplets)
    j = node_js.repeat_interleave(num_triplets)
    k = adj_t_row.storage.col()
    mask = (i != k)
    i, j, k = i[mask], j[mask], k[mask]

    # edge indices (k->j, j->i)
    kj = adj_t_row.storage.value()[mask]
    ji = adj_t_row.storage.row()[mask]

    # calc angles from 0 to pi
    coords_ji = coords[i] - coords[j]
    coords_jk = coords[k] - coords[j]
    a = (coords_ji * coords_jk).sum(dim = -1)
    b = torch.cross(coords_ji, coords_jk).norm(dim = -1) # cross product
    angles = torch.atan2(b, a)

    return dists, angles, node_is, node_js, kj, ji

    
