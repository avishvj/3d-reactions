import os
import numpy as np
from dataclasses import dataclass
from utils.exp import BaseArgs, BaseExpLog

@dataclass
class TSIArgs(BaseArgs):

    # model params
    encoder: str = 'SchNet'
    h_nf: int = 128
    n_layers: int = 2

    # training params
    n_epochs: int = 10
    test_interval: int = 10
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