import os, math
import torch
from dataclasses import asdict

from models.ts_ae.training import construct_tsae, train, test
from utils.exp import construct_logger_and_dir, save_yaml_file
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

    # save experiment params in yaml file
    yaml_file_name = os.path.join(full_log_dir, 'experiment_parameters.yml')
    save_yaml_file(yaml_file_name, asdict(args))

    # data and model
    dataset, train_loader, test_loader = construct_dataset_and_loaders(args)
    tsae, tsae_opt, loss_func = construct_tsae(dataset, args, device)

    # training
    best_test_loss = math.inf
    best_epoch = 0
    exp_log = TSIExpLog(args)

    logger.info("Starting training...")
    for epoch in range(1, args.n_epochs + 1):
        train_loss = train(tsae, train_loader, loss_func, tsae_opt)
        logger.info("Epoch {}: Training Loss {}".format(epoch, train_loss))

        if epoch % args.test_interval == 0:
            test_loss, test_log = test(tsae, test_loader, loss_func)
            logger.info("Epoch {}: Test Loss {}".format(epoch, test_loss))       
            if test_loss <= best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
                torch.save(tsae.state_dict(), os.path.join(full_log_dir, 'best_model.pt'))
            exp_log.add_test_log(test_log)

    logger.info("Best Test Loss {} on Epoch {}".format(best_test_loss, best_epoch))
    exp_log.completed = True

    return exp_log









