import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from rdkit import Chem
from dataclasses import dataclass
from utils.exp import BaseArgs, BaseExpLog

@dataclass
class TSGenArgs(BaseArgs):

    # model params, default set to best MIT model params
    h_nf: int = 256
    gnn_depth: int = 3
    n_layers: int = 2

    # training params
    num_workers: int = 2
    loss: str = 'mse'
    optimiser: str = 'adam' 
    lr: float = 1e-3


class TSGenExpLog(BaseExpLog):

    def __init__(self, args):
        super(TSGenExpLog, self).__init__(args)
    
    def save_Ws(self, file_name, save_to_log_dir = False, W_folder='experiments/meta_eval/ws/'):
        test_Ws = np.concatenate(self.test_logs[-1].Ws, 0).squeeze() # final test log, and remove singleton dims; new dim = num_rxns x 21 x 21
        assert len(test_Ws) == 842, f"Should have 842 test_Ws when unbatched, you have {len(test_Ws)}."
        np.save(W_folder + file_name, test_Ws)
        if save_to_log_dir:
            np.save(self.args.log_dir + 'W', test_Ws)



# recording d_inits

def all_same(items):
    return all(x == items[0] for x in items)

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

NUM_STD_DS = 3

def ensemble_plot(ds_dict, ds_not_to_print, print_my_ds = False):
    num_my_ds = len(ds_dict) - NUM_STD_DS
    ens_ds = []
    for i in range(len(ds_dict['mit'][0])):
        ens_d = 0
        for j in range(0, num_my_ds):
            ens_d += ds_dict[f'D_init{j}'][0][i]
        ens_d /= num_my_ds
        ens_ds.append(ens_d)
    ds_dict['ens'] = (ens_ds, "Avg Ensemble D_init")

    if not print_my_ds:
        for j in range(0, num_my_ds):
            ds_not_to_print.append(f'D_init{j}')

    plot_ds(ds_dict, ds_not_to_print, None)