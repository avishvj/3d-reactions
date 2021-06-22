# data_utils.py, experiment_logger.py

### DATA_UTILS.PY

from torch_geometric.data import Data, Batch

## something to come back to later if necessary
# would have to create CustomDataLoader, CustomBatch, CustomCollater
# then create my own collate() and Batch.from_data_list() funcs
# CustomDataLoader is super simple, the main logic is in CustomCollater which defines the collate() func for the DL
# left this alone because Batch had a fair few funcs and didn't want to cause issues
#   - I think the simple batch each thing works anyway

class CustomBatch(Data):
    def __init__(self, batch = None, ptr):
        pass


def identity_collate(data_list):
    return data_list

class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size = 1, shuffle = False, collate_fn = identity_collate, **kwargs):
        super(CustomDataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn = identity_collate, **kwargs)
        # change to collate_fn = CustomCollater(follow_batch, exclude_keys)
    

class CustomCollater(object):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
    
    def __call__(self, batch):
        return self.collate(batch)
    
    def collate(self, batch):
        elem = batch[0]

        # we have Data(4 torch.Tensor and one int)

        if isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)


        if isinstance(elem, torch.Tensor):
            pass 


### EXPERIMENT_LOGGER.PY

from dataclasses import dataclass
from typing import List

@dataclass
class ExperimentLog:
    # track metrics from experiment

    # train:test split ID?
    # angle changes? and expected: RMSD
    # adj matrix predictions + differences: BCE
    # D_init vs D_final
    # associated gnn_embedding and how it changes
    bce: float

class ExperimentLogs:
    # for experiments over multiple iterations

    def __init__(self):
        self.logs: List[ExperimentLog] = []
    
    def add_exp(self, log: ExperimentLog):
        self.logs.append(log)
    
    def get_ind_metric(self):
        bces = [log.bce for log in self.logs]
        return bces

    def compute_distr(self):
        # compute disrt over metric results for multiple experiments
        bces = self.get_ind_metric()

        # maximum, mminimum, median, mean, std of bces

        # for me: plot dist matrix distributions