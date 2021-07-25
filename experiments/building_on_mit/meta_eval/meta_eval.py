import re
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from ts_vae.data_processors.ts_gen_processor import TSGenDataset
from experiments.building_on_mit.meta_eval.ts_gen_pt.G2C import G2C, train_g2c_epoch
from experiments.exp_utils import ExperimentLog
import numpy as np

# ablation study, UQ i.e. stability testing, weights

class TSGen_ExpLog(ExperimentLog):
    # tsgen: save D_init, W, X, loss for each batch in epoch
    def __init__(self, num_rxns, tt_split, batch_size, epochs, test_interval, recorded_batches = []):
        super(TSGen_ExpLog, self).__init__(num_rxns, tt_split, batch_size, epochs, test_interval, recorded_batches)
        self.epoch_ae_results = []
    
    def add_epoch_result(self, res):
        self.epoch_ae_results.append(res)

def ablation_experiment(tt_split = 0.8, batch_size = 5, epochs = 20, test_interval = 5, \
                        h_nf = 100, n_layers = 2, gnn_depth = 3):
    # ablation study:
    # 1) drop recon_layer, train on D_init, need to create D_GT from X_GT
    # 2) loss with coords and/or D
    # 3) use recon (a) init (b) opt
    # 4) change W used in pytorch, pull init out
    # 5) add noise to training coords and run NLS for diff D_inits

    torch.set_printoptions(precision = 3, sci_mode = False)

    # data prep
    rxns = TSGenDataset(r'data')
    num_rxns = len(rxns)
    num_train = int(np.floor(tt_split * num_rxns))
    train_loader = DataLoader(rxns[: num_train], batch_size = batch_size)
    test_loader = DataLoader(rxns[num_train: ], batch_size = batch_size)

    # model and opt, NOTE: edge_attr.size(2)
    in_node_nf, in_edge_nf = train_loader.dataset[0].x.size(1), train_loader.dataset[0].edge_attr.size(1)
    g2c = G2C(in_node_nf, in_edge_nf, h_nf, n_layers, gnn_depth)
    g2c_opt = torch.optim.Adam(g2c.parameters(), lr = 1e-4)

    experiment_params = (num_rxns, tt_split, batch_size, epochs, test_interval)
    model_params = (g2c, g2c_opt)
    loaders = (train_loader, test_loader)

    print("Starting ablation experiment...")
    train_log, test_log = ablation(experiment_params, model_params, loaders)
    print("Completed ablation experiment, use the experiment log to print results.")

    return train_log, test_log


def ablation(experiment_params, model_params, loaders):

    num_rxns, tt_split, batch_size, epochs, test_interval = experiment_params
    g2c, g2c_opt = model_params
    train_loader, test_loader = loaders

    # log training and testing
    train_log = TSGen_ExpLog(*experiment_params)
    test_log = TSGen_ExpLog(*experiment_params)

    for epoch in range(1, epochs + 1):
        
        train_loss, train_epoch_res = train_g2c_epoch(g2c, g2c_opt, train_loader, test = False)
        if epoch == epochs: # only add final epoch res
            train_log.add_epoch_result(train_epoch_res)
        print(f"===== Training epoch {epoch:03d} complete with loss: {train_loss:.4f} ====")

        if epoch % test_interval == 0:
            test_loss, test_epoch_res = train_g2c_epoch(g2c, g2c_opt, test_loader, test = True)
            test_log.add_epoch_result(test_epoch_res)
            print(f"===== Testing epoch {epoch:03d} complete with loss: {test_loss:.4f} ====")

    return train_log, test_log



