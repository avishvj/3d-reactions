# process as before, but batch-wise
# classes needed:
#   - ReactionTriple([R, TS, P])
#   - ReactionDataset([ReactionTriple])
#       - have get_tracker: factory method for wandb tracker log [ref pyg 3d]
# functions needed:
#   - preprocess raw data
#   - collate(): collate data into batch format
#       - collate_fn specifies how exactly samples need to be batched
#       - collate_fn receives your __getitem__ func
#   - some sort of dataloader_creation()

import torch
import torch.nn.Functional as F
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm

# pyg
from torch_geometric.data import Data #, InMemoryDataset , DataLoader


# in collate.py file

def collate_fn(rxn_batch):
    # param rxn_batch: list of ReactionTriples to be collated

    batch 

    # collates datapoints into batch format
    # returns dict of pytorch tensors



    r_dict = {"nodes": V, "edges": E, "sizes": sizes, "coords": coords}


    # have list of reaction triples that you want to pass in

    # stack tensors in batches
    {torch.nn.utils.rnn.pad_sequence(mol[prop] for mol in rxn_batch) for prop in rxn_batch[0].keys()}


    for i, rxn in enumerate(rxn_batch):
        r, ts, p = rxn_batch[i]


    return



### in main file

class ReactionTriple():
    def __init__(self, r, ts, p):    
        
        super(ReactionTriple, self).__init__()
        
        # all molecules
        self.r = r
        self.ts = ts
        self.p = p

        # number of atoms should be same
        assert len(r.z) == len(ts.z) == len(p.z)
        self.num_atoms = len(r.z)

        # TODO: other checks?

    # TODO: functions to look at reaction core, data vis

class ReactionDataset(Dataset):
    # contains triples of reactant, ts, product
    # PyG could be better but run into issues with collate func

    TYPES = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    TEMP_MOLS_LIMIT = 30

    def __init__(self, root_folder, transform = None, pre_transform = None):

        self.rxn_data = None
        super(ReactionDataset, self).__init__(root_folder, transform, pre_transform)

        self.r_data, self.r_slices = torch.load(self.processed_paths[0])
        self.ts_data, self.ts_slices = torch.load(self.processed_paths[1]) 
        self.p_data, self.p_slices = torch.load(self.processed_paths[2])

    def __init__(self, root_folder, normalise = False, shuffle = False):
        
        self.root_folder = root_folder

        ### need to sort after this point

        self.data = self.process()
        self.num_rxns = len(self.data)

        

        # need a collate function for each batch -> luckily reactants and products are the same length so maybe can hack way round this

        if shuffle:
            self.perm = torch.randperm(self.num_rxns)[:self.num_rxns]
        else:
            self.perm = None
        
    def __len__(self):
        return self.num_rxns

    def __getitem__(self, idx):
        # this is what batches are created from
        return self.data[idx]
    
    def process(self):

        # concat original train and test reactants
        reactants = []
        reactants = self.process_geometry_file('/raw/train_reactants.sdf', reactants)
        reactants = self.process_geometry_file('/raw/test_reactants.sdf', reactants)

        # concat train and test ts
        tss = []
        tss = self.process_geometry_file('/raw/train_ts.sdf', tss)
        tss = self.process_geometry_file('/raw/test_ts.sdf', tss) 

        # concat train and test products
        products = []
        products = self.process_geometry_file('/raw/train_products.sdf', products)
        products = self.process_geometry_file('/raw/test_products.sdf', products) 

        assert len(reactants) == len(tss) == len(products)

        rxns = []
        for rxn_id in range(len(reactants)):
            rxn = ReactionTriple(reactants[rxn_id], tss[rxn_id], products[rxn_id])
            rxns.append(rxn)
        
        return reactants, tss, products, rxns

    def process_geometry_file(self, geometry_file, current_list = None):
        # from QM9 dataset creation

        data_list = current_list if current_list else []
        counted = len(data_list)
        full_path = self.root + geometry_file
        geometries = Chem.SDMolSupplier(full_path, removeHs = False, sanitize = False)

        # get atom and edge features for each geometry
        for i, mol in enumerate(tqdm(geometries)):

            # temp_soln cos of memory issues TODO: change
            if i == self.TEMP_MOLS_LIMIT:
                break

            N = mol.GetNumAtoms()
            # get atom positions as matrix w shape [num_nodes, num_dimensions] = [num_atoms, 3]
            atom_data = geometries.GetItemText(i).split('\n')[4: 4+N]
            atom_positions = [[float(x) for x in line.split()[:3]] for line in atom_data]
            atom_positions = torch.tensor(atom_positions, dtype = torch.float)
            # all the features
            type_idx, atomic_number, aromatic = [], [], []
            sp, sp2, sp3 = [], [], []
            num_hs = []

            # atom/node features
            for atom in mol.GetAtoms():
                type_idx.append(self.TYPES[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridisation = atom.GetHybridization()
                sp.append(1 if hybridisation == HybridizationType.SP else 0)
                sp2.append(1 if hybridisation == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridisation == HybridizationType.SP3 else 0)
                # TODO: lucky does: whether bonded, 3D_rbf
            
            # bond/edge features
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                # edge type for each bond type; *2 because both ways
                edge_type += 2 * [self.BONDS[bond.GetBondType()]]
            # edge_index is graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor([row, col], dtype = torch.long)
            edge_type = torch.tensor(edge_type, dtype = torch.long)
            # edge_attr is edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = F.one_hot(edge_type, num_classes = len(self.BONDS)).to(torch.float)

            # order edges based on combined ascending order
            asc_order_perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, asc_order_perm]
            edge_type = edge_type[asc_order_perm]
            edge_attr = edge_attr[asc_order_perm]

            row, col = edge_index
            z = torch.tensor(atomic_number, dtype = torch.long)
            hs = (z == 1).to(torch.float) # hydrogens
            num_hs = scatter(hs[row], col, dim_size = N).tolist() # scatter helps with one-hot

            x1 = F.one_hot(torch.tensor(type_idx), num_classes = len(self.TYPES))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype = torch.float).t().contiguous()
            x = torch.cat([x1.to(torch.float), x2], dim = -1)

            idx = counted + i 
            # TODO: dict instead of Data?
            mol_data = Data(x = x, z = z, pos = atom_positions, edge_index = edge_index, edge_attr = edge_attr, idx = idx)
            data_list.append(mol_data)
        
        return data_list


        



    










