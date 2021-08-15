import os
import torch
from dataclasses import asdict

from utils.exp import construct_logger_and_dir, save_yaml_file
from models.ts_gen_pt.G2C import construct_model_opt_loss
from utils.data import construct_dataset_and_loaders
from utils.ts_interpolation import TSIExpLog

def tsi_experiment(args):

    # construct logger, exp directory, and set device
    torch.set_printoptions(precision = 3, sci_mode = False)
    log_file_name = 'tsi'
    logger, full_log_dir = construct_logger_and_dir(log_file_name, args.log_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # LP: cuda or AV: cuda:0
    args.log_dir = full_log_dir
    args.log_file_name = log_file_name + '.log'
    args.device = device

    # save experiment params in yaml file, NOTE: model params also saved in construct_model_opt_loss()
    yaml_file_name = os.path.join(full_log_dir, 'experiment_parameters.yml')
    save_yaml_file(yaml_file_name, asdict(args))

    # data and model
    dataset, train_loader, test_loader = construct_dataset_and_loaders(args)
    g2c, g2c_opt, loss_func = construct_model_opt_loss(dataset, args, device)







