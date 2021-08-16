from typing import Dict
import numpy as np
from dataclasses import dataclass
from utils.exp import BaseArgs, BaseExpLog
from utils.data import remove_processed_data
from data.data_processors.grambow_processor import ReactionDataset
from torch_geometric.data import DataLoader

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