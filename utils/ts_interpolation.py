import numpy as np
from dataclasses import dataclass
from utils.exp import BaseArgs, BaseExpLog
from utils.data import remove_processed_data
from data.data_processors.grambow_processor import ReactionDataset
from torch_geometric.data import DataLoader

import torch
from torch_geometric.utils import to_dense_batch
from utils.models import X_to_dist
from rdkit import Chem
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

@dataclass
class SchNetParams:
    n_filters: int = 64
    n_gaussians: int = 64
    cutoff_val: float = 10.0

    h_nf: int = 32
    n_layers: int = 2
    device: str = 'cpu'

@dataclass
class EGNNParams:
    in_node_nf: int
    in_edge_nf: int

    h_nf: int = 32
    out_nf: int = h_nf
    emb_nf: int = h_nf
    n_layers: int = 2
    device: str = 'cpu'


@dataclass
class TSIArgs(BaseArgs):
    # model params
    encoder_type: str = 'EGNN'

    # training params
    num_workers: int = 2
    optimiser: str = 'adam' 
    lr: float = 1e-3


class TSIExpLog(BaseExpLog):

    def __init__(self, args):
        super(TSIExpLog, self).__init__(args)
    
    def save_embs(self, file_name, save_to_log_dir = False, emb_folder='experiments/ts_interpolation/embs/'):
        test_embs = np.concatenate(self.test_logs[-1].embs, 0) # final test log, new dim = num_rxns x 21 x 21
        assert len(test_embs) == 842, f"Should have 842 test_D_inits when unbatched, you have {len(test_embs)}."
        np.save(emb_folder + file_name, test_embs)
        if save_to_log_dir:
            np.save(self.args.log_dir + 'emb', test_embs)


def construct_dataset_and_loaders(args):

    if args.remove_existing_data:
        remove_processed_data()
    
    # build dataset
    dataset = ReactionDataset(args.root_dir, args.n_rxns)
    
    # build loaders using tt_split
    n_rxns = len(dataset) # as args.n_rxns may be over the limit
    n_train = int(np.floor(args.tt_split * n_rxns))

    to_follow = ['edge_index_r', 'edge_index_ts', 'edge_index_p', 
                 'edge_attr_r', 'edge_attr_ts', 'edge_attr_p', 
                 'pos_r', 'pos_ts', 'pos_p', 
                 'x_r', 'x_ts', 'x_p',
                 'z_r', 'z_ts', 'z_p']
    train_loader = DataLoader(dataset[: n_train], batch_size = args.batch_size, follow_batch = to_follow, \
        shuffle = True, num_workers = args.num_workers, pin_memory = True)
    test_loader = DataLoader(dataset[n_train:], batch_size = args.batch_size, follow_batch = to_follow, \
        shuffle = False, num_workers = args.num_workers, pin_memory = True)
    
    return dataset, train_loader, test_loader


def check_test_distribution(args):
    """Check test distribution of TS reaction core is same as original MIT work."""
    
    mols_folder = r'data/raw/'

    reactant_file = mols_folder + 'test_reactants.sdf'
    test_r = Chem.SDMolSupplier(reactant_file, removeHs=False, sanitize=False)
    test_r = [x for x in test_r]

    product_file = mols_folder + 'test_products.sdf'
    test_p = Chem.SDMolSupplier(product_file, removeHs=False, sanitize=False)
    test_p = [x for x  in test_p]

    test_ts_file = mols_folder + 'test_ts.sdf'
    test_ts = Chem.SDMolSupplier(test_ts_file, removeHs=False, sanitize=False)
    test_ts = [ts for ts in test_ts]

    _, _, test_loader = construct_dataset_and_loaders(args)

    D_gts = []
    for idx, rxn_batch in enumerate(test_loader):
        X_gt, mask = to_dense_batch(rxn_batch.pos_ts, rxn_batch.x_ts_batch, 0., max(rxn_batch.num_atoms)) # pos_ts = [b * max_num_nodes, 3]
        batched_D_gt = X_to_dist(X_gt)
        batched_D_gt = [x[-1] for x in torch.split(batched_D_gt, 1, 0)]
        D_gts.extend(batched_D_gt)

    assert len(D_gts) == 842, f"Number of test dist matrices is {len(D_gts)}, you need 842 which was originally published in the MIT model."

    mine, gt = [], []
    for idx in range(len(test_ts)):

        # num_atoms + mask for reaction core
        num_atoms = test_ts[idx].GetNumAtoms()
        core_mask = (Chem.GetAdjacencyMatrix(test_p[idx]) + Chem.GetAdjacencyMatrix(test_r[idx])) == 1

        gt.append(np.ravel(Chem.Get3DDistanceMatrix(test_ts[idx]) * core_mask))
        mine.append(np.ravel(D_gts[idx][0:num_atoms, 0:num_atoms] * core_mask))
        
    all_ds = [gt, mine]
    all_ds = [np.concatenate(ds).ravel() for ds in all_ds]
    all_ds = [ds[ds != 0] for ds in all_ds] # only keep non-zero values

    fig, ax = plt.subplots(figsize=(12,9))
    sns.distplot(all_ds[0], color='b', kde_kws={"lw": 5, "label": "GT"}, hist=False)
    sns.distplot(all_ds[1], color='r', kde_kws={"lw": 3, "label": "Mine"}, hist=False)

    ax.legend(loc='upper right')
    ax.legend(fontsize=12)
    ax.set_ylabel('Density', fontsize=22)
    ax.set_xlabel(r'Distance ($\AA$)', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(True)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(True) 