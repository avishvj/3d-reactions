import os, time, logging, yaml
import numpy as np
import matplotlib.pyplot as plt

### logging

def construct_logger_and_dir(log_file_name, log_dir = 'log', exp_dir = None) -> logging.Logger:
    # NOTE: exp_dir needs / at end
    
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # set logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if exp_dir is None:
        exp_dir = time.strftime("%y%b%d_%I%M%p/", time.localtime())
    full_log_dir = os.path.join(log_dir, exp_dir)

    if not os.path.exists(full_log_dir):
        os.makedirs(full_log_dir)
    
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
        fig.savefig(os.path.join(os.path.dirname(log_file), 'tt_loss.pdf'), bbox_inches='tight')