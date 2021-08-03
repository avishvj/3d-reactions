from dataclasses import dataclass

@dataclass
class TSGenArgs:
    # from dataclasses import asdict when needed

    # logistics params
    root_dir: str = r'data'
    log_dir: str = r'log'
    verbose: bool = True
    remove_existing_data: bool = False
    
    # data params
    n_rxns: int = 8000 # if over limit, takes max possible ~7600
    tt_split: float = 0.889 # to return 842 like MIT
    batch_size: int = 8
    
    # model params
    h_nf: int = 32
    gnn_depth: int = 3
    n_layers: int = 2

    # training params
    n_epochs: int = 10
    test_interval: int = 10
    num_workers: int = 2
    loss: str = 'mse'
    optimiser: str = 'adam' 
    lr: float = 1e-3