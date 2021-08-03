import os, time, logging, yaml
import numpy as np
import matplotlib.pyplot as plt

### logging

def construct_logger_and_dir(log_file_name, log_dir = 'log', exp_dir = None) -> logging.Logger:
    # NOTE: exp_dir needs / at end
    
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # set logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if exp_dir is None:
        exp_dir = time.strftime("%y%b%d_%I%M%p/", time.localtime())
    full_log_dir = os.path.join(log_dir, exp_dir)

    if not os.path.exists(full_log_dir):
        os.makedirs(full_log_dir)
    
    fh = logging.FileHandler(os.path.join(full_log_dir, log_file_name + '.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger, full_log_dir

def save_yaml_file(path, content):
    if not isinstance(path, str):
        raise InputError(f'Path must be a string, got {path} which is a {type(path)}')
    yaml.add_representer(str, string_representer)
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)

def string_representer(dumper, data):
    """Add a custom string representer to use block literals for multiline strings."""
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)

### plotting

def plot_tt_loss(log_file, save_fig = False):
    train_loss = []
    test_loss = []
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if ': Training Loss' in line:
                train_loss.append(float(line.split(' ')[-1].rstrip()))
            if ': Test Loss' in line:
                test_loss.append(float(line.split(' ')[-1].rstrip()))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(np.arange(len(train_loss)), train_loss, label='Train Loss')
    ax.plot(np.arange(len(test_loss)), test_loss, label='Test Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    if save_fig:
        fig.savefig(os.path.join(os.path.dirname(log_file), 'tt_loss.pdf'), bbox_inches='tight')


### old stuff, TODO: remove?

from dataclasses import dataclass
from typing import List, Tuple
import torch.tensor as Tensor

@dataclass
class BatchLog:
    # create this every time you end a batch 

    # batch metadata
    batch_size: int
    batch_counter: int
    num_nodes: int
    batch_node_vecs: Tuple[Tensor, Tensor]

    # losses
    coord_loss: float
    adj_loss: float
    node_loss: float
    batch_loss: float

    # dynamics: coord, adj
    record_dynamics: bool = False

    if record_dynamics:
        # angles? could do that after getting coords
        # or maybe record actual D_init [your D_final is last iteration of batch]
        # record embeddings
        coords: Tensor
        adjs: Tensor
        record_id: int = batch_counter 


class EpochLog:

    def __init__(self):

        # batch log
        self.batch_logs: List[BatchLog] = []

        # epoch metadata
        # self.batch_size = batch_size
    
    def add_batch(self, log: BatchLog):
        self.batch_logs.append(log)
    
    def get_epoch_losses(self):
        # TODO: should get some sort of overall loss

        adj_losses = [bl.adj_loss for bl in self.batch_logs]
        coord_losses = [bl.coord_loss for bl in self.batch_logs]
        node_losses = [bl.node_loss for bl in self.batch_logs]
        batch_losses = [bl.batch_loss for bl in self.batch_logs]
        return adj_losses, coord_losses, node_losses, batch_losses

    def get_recorded_batches(self):
        # all the batch_ids that were chosen to record in a list
        # TODO: sort this out

        recorded_batches = []

        for batch_log in self.batch_logs:
            if batch_log.record_dynamics:
                recorded_batches.append(batch_log.record_id)

        # [b_log.record_id for b_log in self.batch_logs if batch_log.record_dynamics]

        return recorded_batches
    

class ExperimentLog:
    """ For tracking key stats over multiple epochs. """

    # rep metadata: init type, featurise operator, combination operator

    # experiment metadata: num_rxns, tt_split, batch_size
    # associated gnn_embedding and how it changes

    def __init__(self, num_rxns, tt_split, batch_size, epochs, test_interval, recorded_batches):
        
        self.epoch_logs: List[EpochLog] = []
        
        # experiment metadata
        self.num_rxns = num_rxns
        self.tt_split = tt_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_interval = test_interval
        # batches to record
        if recorded_batches and recorded_batches != []:
            assert max(recorded_batches) < (num_rxns / batch_size), "Maximum batch_id not possible to record"
            assert all(batch_id > -1 for batch_id in recorded_batches), "You are planning to record negative batch ids."
        self.recorded_batches = recorded_batches

    def add_epoch(self, epoch_log):
        self.epoch_logs.append(epoch_log)
    
    def get_performed_epochs(self):
        self.performed_epochs = len(self.epoch_logs)
        return self.performed_epochs
    
    def get_experiment_losses(self):
        # for each epoch, record the losses
        
        experiment_adj_losses = []
        experiment_coord_losses = []
        experiment_node_losses = []
        experiment_batch_losses = []

        for epoch_log in self.epoch_logs:
            adj_losses, coord_losses, node_losses, batch_losses = epoch_log.get_epoch_losses()
            experiment_adj_losses.append(adj_losses)
            experiment_coord_losses.append(coord_losses)
            experiment_node_losses.append(node_losses)
            experiment_batch_losses.append(batch_losses)
        
        return experiment_adj_losses, experiment_coord_losses, experiment_node_losses, experiment_batch_losses
    
    def get_batch_dynamics(self):

        coord_dynamics = []
        adj_dynamics = []

        for batch_id in self.recorded_batches:
            assert self.epoch_logs.batch_logs[batch_id].record_dynamics, f"This batch isn't set to record! ID: {batch_id}"
            batch = self.epoch_logs.batch_logs[batch_id]
            coord_dynamics.append(batch.coords)
            adj_dynamics.append(batch.adjs)
            
        return coord_dynamics, adj_dynamics  

    def compute_distributions(self):
        # get individual metrics
        a_losses, c_losses, n_losses, b_losses = self.get_experiment_losses()
        # maximum, mminimum, median, mean, std of metrics
        pass