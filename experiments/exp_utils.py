from dataclasses import dataclass
from typing import List
import torch.tensor as Tensor

# experiment metadata: number of molecules, epochs, optimiser 

# best_epoch_train_res: epoch number, loss, embeddings
# best_epoch_test_res: epoch number, loss, embeddings

# might need to create experiment classes for each of these

# Building on the MIT model
#   - 'Is the model learning?' experiment
#       - Standard loss curve comparison: evaluate each individual component
#           - Map embeddings to what is learnt a la GraphVAE?
#       - Scatter plot between train:test in embedding space and molecular space
#       - Coord evolution dynamics [as distribution?] [to deal with diff numbers of atoms, could pick sample]
#       - Adj evolution dyanmics [as distribution?] [sample?]
#   - 'What is the model learning?' experiment
#       - Fig 3: look at weights corresponding to bonds
#           - MY equivalent could be with attention module?
#       - 'TS interpolation' experiment:
#           - PCA on R and P embeddings; compare their linear comb embedding to TS embedding         
#   - 'Uncertainty quantification' experiment
#       - Fig 2: reaction core distances, compare 4 MIT to mine
#           - Note: currently, I have D_init from my model as coord_out
#           - If not D_init distribution not multimodal, motivate work through rep learning i.e. ideal feature mappings and TS operator
#       - Bayesian version of model to get uncertainty estimates, aiming for robustness of HPs
#           - Simple, foolproof methods: DeepEnsemble, dropout
#           - Advanced: subspace inference method

# TS generation as a representation learning problem
#   - Ideal molecule representation e.g. interatomic distances, etc.
#   - Feature mapping comparisons
#       - How to compare embeddings?
#   - TS operator comparisons
#       - Best stage to concat/average
#       - Best function to use

# TODO
#   - My unique loss functions here?

@dataclass
class BatchLog:
    # create this every time you end a batch 

    # batch metadata
    batch_size: int
    batch_counter: int
    num_nodes: int

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

    def __init__(self, num_rxns, tt_split, batch_size, recorded_batches):
        
        self.epoch_logs: List[EpochLog] = []
        
        # experiment metadata
        self.num_rxns = num_rxns
        self.tt_split = tt_split
        self.batch_size = batch_size
        # batches to record
        assert max(recorded_batches) < (num_rxns / batch_size), "Maximum batch_id not possible to record"
        assert all(batch_id > -1 for batch_id in recorded_batches), "You are planning to record negative batch ids."
        self.recorded_batches = recorded_batches

    
    def add_epoch(self, epoch_log):
        self.epoch_logs.append(epoch_log)
    
    def get_num_epochs(self):
        self.num_epochs = len(self.epoch_logs)
    
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