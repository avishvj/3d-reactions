import os, math, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import torch
from rdkit import Chem
from dataclasses import asdict

from data.data_processors.new_ts_gen_processor import construct_dataset_and_loaders
from experiments.exp_utils import construct_logger_and_dir, plot_tt_loss, save_yaml_file
from experiments.building_on_mit.meta_eval.ts_gen_pt.G2C import construct_model_opt_loss
from experiments.building_on_mit.meta_eval.ts_gen_pt.training import train, test

# TODO: ablation study, UQ i.e. stability testing, weights

def experiment(args, plot=False):

    # construct logger, exp directory, and set device
    torch.set_printoptions(precision = 3, sci_mode = False)
    log_file_name = 'train'
    logger, full_log_dir = construct_logger_and_dir(log_file_name, args.log_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # LP: cuda or AV: cuda:0
    args.device = device

    # save experiment params in yaml file, NOTE: model params also saved in construct_model_opt_loss()
    yaml_file_name = os.path.join(full_log_dir, 'experiment_parameters.yml')
    save_yaml_file(yaml_file_name, asdict(args))

    # data and model
    dataset, train_loader, test_loader = construct_dataset_and_loaders(args)
    g2c, g2c_opt, loss_func = construct_model_opt_loss(dataset, args, full_log_dir, device)
    
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

### plotting ds

def save_d(file_name, all_test_res, d_folder='d_inits/'):
    # create and save D_init of [num_train, 21, 21] for plotting
    batched_test_Ds = all_test_res[-1]['D_pred'] 
    test_Ds = np.concatenate([D for D in batched_test_Ds], 0) # expand along batches
    assert len(test_Ds) == 842, f"Should have 842 test_D_inits when unbatched, you have {len(test_Ds)}."
    np.save(d_folder + file_name, test_Ds)

def create_ds_dict(d_files, d_folder='d_inits/', mols_folder=r'data/raw/'):
    # base_folder is where the test mol sdf files are
    # all_test_res is dict of D_preds, TODO: add assert
    # TODO: add way to automate loading multiple files ... pass in file names

    # get test mols
    test_ts_file = mols_folder + 'test_ts.sdf'
    reactant_file = mols_folder + 'test_reactants.sdf'
    product_file = mols_folder + 'test_products.sdf'
    test_r = Chem.SDMolSupplier(reactant_file, removeHs=False, sanitize=False)
    test_r = [x for x in test_r]
    test_ts = Chem.SDMolSupplier(test_ts_file, removeHs=False, sanitize=False)
    test_ts = [ts for ts in test_ts]
    test_p = Chem.SDMolSupplier(product_file, removeHs=False, sanitize=False)
    test_p = [x for x  in test_p]

    # save and load
    mit_d_init = np.load(d_folder + 'mit_best.npy')
    d_inits = []
    for d_file in d_files:
        d_inits.append(np.load(d_folder + d_file))
    num_d_inits = len(d_inits)
    
    # lists for plotting
    gt, mit, lin_approx = [], [], []
    d_init_lists = [[] for _ in range(num_d_inits)]

    for idx in range(len(test_ts)):

        # num_atoms + mask for reaction core
        num_atoms = test_ts[idx].GetNumAtoms()
        core_mask = (Chem.GetAdjacencyMatrix(test_p[idx]) + Chem.GetAdjacencyMatrix(test_r[idx])) == 1

        # main 3
        gt.append(np.ravel(Chem.Get3DDistanceMatrix(test_ts[idx]) * core_mask))
        mit.append(np.ravel(mit_d_init[idx][0:num_atoms, 0:num_atoms] * core_mask))
        lin_approx.append(np.ravel((Chem.Get3DDistanceMatrix(test_r[idx]) + Chem.Get3DDistanceMatrix(test_p[idx])) / 2 * core_mask))

        # other d_inits
        for j, d_init_list in enumerate(d_init_lists):
            d_init_lists[j].append(np.ravel(d_inits[j][idx][0:num_atoms, 0:num_atoms]*core_mask))
    
    # make plottable
    all_ds = [gt, mit, lin_approx, *d_init_lists]
    all_ds = [np.concatenate(ds).ravel() for ds in all_ds]
    assert all_same([len(ds) for ds in all_ds]), "Lengths of all ds after concat don't match."
    all_ds = [ds[ds != 0] for ds in all_ds] # only keep non-zero values
    assert all_same([len(ds) for ds in all_ds]), "Lengths of all ds after removing zeroes don't match."

    ds_dict = {'gt': (all_ds[0], 'Ground Truth'), 'mit': (all_ds[1], 'MIT D_init'), \
               'lin_approx': (all_ds[2], 'Linear Approximation')}
    base_ds_counter = len(ds_dict)
    
    for d_id in range(len(d_init_lists)):
        name = f'D_init{d_id}'
        ds_dict[name] = (all_ds[base_ds_counter + d_id], name)
    
    return ds_dict

def plot_ds(ds_dict, no_print=[], save_fig_name=None):
    # keys: 'gt', 'lin_approx', 'mit', f'D_init{i}'

    fig, ax = plt.subplots(figsize=(12,9))
    num_to_plot = len(ds_dict)
    cols = sns.color_palette("Set2", num_to_plot)

    # print for all keys not in no_print
    for i, key in enumerate(ds_dict.keys()):
        if key in no_print:
            continue
        sns.distplot(ds_dict[key][0], color=cols[i], kde_kws={"lw": 3, "label": ds_dict[key][1]}, hist=False)

    ax.legend(loc='upper right')
    ax.legend(fontsize=12)
    ax.set_ylabel('Density', fontsize=22)
    ax.set_xlabel(r'Distance ($\AA$)', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(True)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(True) 

    if save_fig_name:
        plt.savefig(f'{save_fig_name}.png', bbox_inches='tight')

def all_same(items):
    return all(x == items[0] for x in items)