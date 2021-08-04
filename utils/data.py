import glob, os, numpy as np
from torch_geometric.data import DataLoader
from data.data_processors.new_ts_gen_processor import TSGenDataset

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