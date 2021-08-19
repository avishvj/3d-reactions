import os, time, logging, yaml
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class BaseArgs:
    # from dataclasses import asdict when needed

    # logistics params
    root_dir: str = r'data'
    log_dir: str = r'log'
    log_file_name: str = r'train'
    verbose: bool = True
    remove_existing_data: bool = False
    
    # data params
    n_rxns: int = 8000 # if over limit, takes max possible ~7600
    tt_split: float = 0.889 # to return 842 test like MIT
    batch_size: int = 8 # default set to best MIT model params
    
    # training params
    n_epochs: int = 10
    test_interval: int = 10

class BaseExpLog:
    def __init__(self, args):
        self.args = args
        self.test_logs = []
        self.completed = False
    
    def add_test_log(self, test_log):
        self.test_logs.append(test_log)
    
    def save_Ds(self, file_name, D_folder='experiments/meta_eval/d_inits/', save_to_log_dir = False):
        """Allows for saving distance matrices from testing model in folder and log file."""
        assert self.check_test_batches(self.test_logs[-1].Ds), "You don't have the same number of batched D files as batches."
        test_Ds = np.concatenate(self.test_logs[-1].Ds, 0) # final test log, new dim = num_rxns x 21 x 21
        assert len(test_Ds) == 842, f"Should have 842 test_D_inits when unbatched, you have {len(test_Ds)}."
        np.save(D_folder + file_name, test_Ds)
        if save_to_log_dir:
            np.save(self.args.log_dir + 'D' + file_name, test_Ds)

    def plot_loss(self, save_fig=False):
        if not self.completed:
            raise Exception("Experiment has not been run yet.")
        log_file_path = os.path.join(os.path.dirname(self.args.log_dir), self.args.log_file_name)
        plot_tt_loss(log_file_path, save_fig)
    
    def check_test_batches(self, test_log_files):
        n_train = int(np.floor(self.args.tt_split * self.args.n_rxns))
        n_test = self.args.n_rxns - n_train
        n_full_batches = n_test // self.args.batch_size
        n_partial_batches = 0 if n_test % self.args.batch_size == 0 else 1
        n_total_batches = n_full_batches + n_partial_batches
        return len(test_log_files) == n_total_batches


class TestLog:
    def __init__(self):
        self.Ds = []
        self.Ws = []
        self.embs = []
    
    def add_D(self, D):
        """Adds batch of Ds."""
        self.Ds.append(D)
    
    def add_W(self, W):
        """Adds batch of Ws."""
        self.Ws.append(W)

    def add_emb(self, emb):
        """Adds batch of node embeddings for R/P/TS."""
        self.embs.append(emb)

### logging

def construct_logger_and_dir(log_file_name, log_dir = 'log', exp_dir = None) -> logging.Logger:
    # NOTE: exp_dir needs / at end
    
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # set up console logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if exp_dir is None:
        exp_dir = time.strftime("%y%b%d_%I%M%p/", time.localtime())
    full_log_dir = os.path.join(log_dir, exp_dir)

    if not os.path.exists(full_log_dir):
        os.makedirs(full_log_dir)
    
    # set up file logging
    fh = logging.FileHandler(os.path.join(full_log_dir, log_file_name + '.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger, full_log_dir

def save_yaml_file(path, content):
    if not isinstance(path, str):
        raise InputError(f'Path must be a string, got {path} which is a {type(path)}')
    yaml.add_representer(str, string_representer)
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)

def string_representer(dumper, data):
    """Add a custom string representer to use block literals for multiline strings."""
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)

### plotting

def plot_tt_loss(log_file, save_fig = False):
    train_loss = []
    test_loss = []
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if ': Training Loss' in line:
                train_loss.append(float(line.split(' ')[-1].rstrip()))
            if ': Test Loss' in line:
                test_loss.append(float(line.split(' ')[-1].rstrip()))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(np.arange(len(train_loss)), train_loss, label='Train Loss')
    ax.plot(np.arange(len(test_loss)), test_loss, label='Test Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    if save_fig:
        fig.savefig(os.path.join(os.path.dirname(log_file), 'tt_loss.png'), bbox_inches='tight')