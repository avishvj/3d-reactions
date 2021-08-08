import glob, os, numpy as np
from torch_geometric.data import DataLoader
from data.data_processors.ts_gen_processor import TSGenDataset

def construct_dataset_and_loaders(args):

    if args.remove_existing_data:
        remove_processed_data()
    
    # build dataset
    dataset = TSGenDataset(args.root_dir, args.n_rxns)
    
    # build loaders using tt_split
    n_rxns = len(dataset) # as args.n_rxns may be over the limit
    n_train = int(np.floor(args.tt_split * n_rxns))
    train_loader = DataLoader(dataset[: n_train], batch_size = args.batch_size, \
        shuffle = True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(dataset[n_train: ], batch_size = args.batch_size, \
        shuffle = False, num_workers=args.num_workers, pin_memory=True)
    
    return dataset, train_loader, test_loader

def remove_processed_data():
    files = glob.glob(r'data/processed/*')
    for f in files:
        os.remove(f)
    print("Files removed.")


### dataset perturbations
# TODO: weight perturbations in similar way?

def construct_perturbed_dataset_and_loaders(args, perturbation):
    # NOTE: this perturbation will be different to mine because of diff init
    # TODO: perturb sdf or perturb constructed dataset?

    if args.remove_existing_data:
        remove_processed_data()
    
    # build dataset
    dataset = TSGenDataset(args.root_dir, args.n_rxns)

    for ts_init in dataset:
        ts_init = perturb_ts(ts_init, perturbation)
    
    return dataset
    
import torch

def perturb_ts(ts_init, perturbation = 'std_norm'):
    # perturbation = 'noise' or 'rotate'

    if perturbation == 'std_norm':
        # add standard normally distributed noise to each position 

        noise = torch.randn(1)

        for atom_i in ts_init: # TODO: add to positional info so to edge features then?
            atom_i += noise

    elif perturbation == 'rotate':
        # rotate: find reaction axis, rotate each molecule around that
        
        # axis = 
        # rotate func
        print()

    else:
        raise NotImplementedError("Input perturbation invalid. Try TODO")

    
    return ts_init