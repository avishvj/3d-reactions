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