# UQ
# weights

# ablation study: to do in meta_eval!
# 1) drop recon_layer
# 2) do loss with coords instead of dist matrix?
# 3) use recon (a) init (b) opt
# 4) change W used in pytorch, pull init out
# 5) add noise to training coords and run NLS for diff D_inits



# data processing
from ts_vae.data_processors.ts_gen_processor import TSGenDataset
from torch_geometric.data import DataLoader

# torch, torch geometric
import torch
import torch.nn as nn

# model
from experiments.building_on_mit.meta_eval.ts_gen_pt.G2C import G2C, train
# TODO: get train

# experiment
from experiments.exp_utils import ExperimentLog
import numpy as np

def ablation_experiment(tt_split = 0.8, batch_size = 5, epochs = 20, test_interval = 5):

    torch.set_printoptions(precision = 3, sci_mode = False)

    # data prep
    rxns = TSGenDataset(r'data')
    num_rxns = len(rxns)
    num_train = int(np.floor(tt_split * num_rxns))
    batch_size = 5
    train_loader = DataLoader(rxns[: num_train], batch_size = batch_size)
    test_loader = DataLoader(rxns[num_train: ], batch_size = batch_size)

    # model params

    # model and opt
    g2c = G2C(in_node_nf, in_edge_nf, h_nf, n_layers = 2, num_iterations = 3)
    g2c_opt = torch.optim.Adam(g2c.parameters(), lr = 1e-3)

    # ablation study

    experiment_params = (num_rxns, tt_split, batch_size, epochs, test_interval, recorded_batches)
    model_params = (g2c, g2c_opt)
    loaders = (train_loader, test_loader)

    print("Starting ablation experiment...")
    train_log, test_log = ablation(experiment_params, model_params, loaders)
    print("Completed ablation experiment, use the experiment log to print results.")

    return train_log, test_log

def ablation(experiment_params, model_params, loaders):
    pass



