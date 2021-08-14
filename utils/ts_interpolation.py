import os
import numpy as np
from dataclasses import dataclass
from utils.exp import plot_tt_loss

@dataclass
class TSIArgs:
    # from dataclasses import asdict when needed

    # logistics params
    root_dir: str = r'data'
    log_dir: str = r'log'
    log_file_name: str = r'tsi'
    verbose: bool = True
    remove_existing_data: bool = False
    
    # data params
    n_rxns: int = 8000 # if over limit, takes max possible ~7600
    tt_split: float = 0.889 # to return 842 test like MIT
    batch_size: int = 8
    
    # model params, default set to best MIT model params
    h_nf: int = 128
    # gnn_depth: int = 3
    n_layers: int = 2

    # training params
    n_epochs: int = 10
    test_interval: int = 10
    # num_workers: int = 2
    # loss: str = 'mse'
    optimiser: str = 'adam' 
    lr: float = 1e-3

class ExpLog:
    def __init__(self, args, test_logs=[]):
        self.args = args
        self.test_logs = test_logs
        self.completed = False

    def add_test_log(self, test_log):
        self.test_logs.append(test_log)
    
    def save_embs(self, file_name, save_to_log_dir = False, emb_folder='experiments/ts_interpolation/embs/'):
        test_embs = np.concatenate(self.test_logs[-1].embs, 0) # final test log, new dim = num_rxns x 21 x 21
        assert len(test_embs) == 842, f"Should have 842 test_D_inits when unbatched, you have {len(test_embs)}."
        np.save(emb_folder + file_name, test_embs)
        if save_to_log_dir:
            np.save(self.args.log_dir + 'emb', test_embs)
    
    def save_Ds(self, file_name, save_to_log_dir = False, D_folder='experiments/ts_interpolation/ds/'):
        test_Ds = np.concatenate(self.test_logs[-1].Ds, 0) # final test log, new dim = num_rxns x 21 x 21
        assert len(test_Ds) == 842, f"Should have 842 test_D_inits when unbatched, you have {len(test_Ds)}."
        np.save(D_folder + file_name, test_Ds)
        if save_to_log_dir:
            np.save(self.args.log_dir + 'D', test_Ds)
    
    def plot_loss(self, save_fig=False):
        if not self.completed:
            raise Exception("Experiment has not been run yet.")
        log_file_path = os.path.join(os.path.dirname(self.args.log_dir), self.args.log_file_name)
        plot_tt_loss(log_file_path, save_fig)
