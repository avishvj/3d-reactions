import os, math, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import torch
from rdkit import Chem
from dataclasses import asdict

from utils.data import construct_dataset_and_loaders
from utils.exp import construct_logger_and_dir, plot_tt_loss, save_yaml_file
from models.ts_gen_pt.G2C import construct_model_opt_loss
from models.ts_gen_pt.training import train, test

# TODO: ablation study, UQ i.e. stability testing, weights

def experiment(args, plot=False):

    # construct logger, exp directory, and set device
    torch.set_printoptions(precision = 3, sci_mode = False)
    log_file_name = 'train'
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
    
    # multi gpu training, TODO
    #if torch.cuda.device_count() > 1:
    #    logger.info(f'Using {torch.cuda.device_count()} GPUs for training...')
    #    g2c = torch.nn.DataParallel(g2c)

    # training
    best_test_loss = math.inf
    best_epoch = 0
    all_test_res = []
    
    logger.info("Starting training...")
    for epoch in range(1, args.n_epochs + 1):
        train_loss = train(g2c, train_loader, loss_func, g2c_opt, logger)
        logger.info("Epoch {}: Training Loss {}".format(epoch, train_loss))

        if epoch % args.test_interval == 0:
            test_loss, test_res = test(g2c, test_loader, loss_func, full_log_dir)
            logger.info("Epoch {}: Test Loss {}".format(epoch, test_loss))       
            if test_loss <= best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
                torch.save(g2c.state_dict(), os.path.join(full_log_dir, 'best_model.pt'))
            all_test_res.append(test_res)
    
    logger.info("Best Test Loss {} on Epoch {}".format(best_test_loss, best_epoch))
    log_file = os.path.join(full_log_dir, log_file_name + '.log')
    if plot:
        plot_tt_loss(log_file)

    return all_test_res 