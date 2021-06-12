from rdkit import Chem
import pymol
import tempfile, os
import numpy as np

# what other TS properties do I need? syndags has good code for properties.

# ts_gen have pymol render but one of their imports tempfile doesn't exist anymore?



def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)