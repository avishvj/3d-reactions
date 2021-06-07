from scipy.sparse import data
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
from enum import Enum

# TODO: abstract classes for each type? can that be done with InMemoryDataset?

TEMP_MOLS_LIMIT = 842 * 2

class GeometryFile(Enum):
    """ Enum for indices corresponding to each reactant, transition state, product file. 
        Use this ordering convention throughout the project.
        TODO: give enums their own file.
    """
    train_r = 0
    train_ts = 1
    train_p = 2
    test_r = 3
    test_ts = 4
    test_p = 5

class ConcatGeometryFile(Enum):
    """ Enum for indices corresponding to each reactant, transition state, product file. 
        Use this ordering convention throughout the project.
        TODO: give enums their own file.
    """
    train_concat_rp = 0
    train_ts = 1
    test_concat_rp = 2
    test_ts = 3

class ReactionDataset(InMemoryDataset):
    """ Creates instance of reaction dataset. 
        To add a new file: create file in data/raw, add to GeometryFile enum, add to func raw_file_names(), and add to processed_file_names().
        TODO: remove TEMP_MOLS_LIMIT.
        TODO: change Temp name.
        TODO: hackiness of if/elif in init(), processed_file_names(), process(). 
        TODO: combine train and test into one for easier loading, perhaps?
        TODO: was this necessary? for concatenation, you could just train each geometry file after another, no?
    """
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    def __init__(self, root, geo_file, dataset_type, transform=None, pre_transform=None):
        self.dataset_type = dataset_type
        super(ReactionDataset, self).__init__(root, transform, pre_transform)
        
        self.geo_file = geo_file        

        if dataset_type == 'individual':
            self.data, self.slices = torch.load(self.processed_paths[GeometryFile[geo_file].value]) # ind
        
        elif dataset_type == 'concat':
            self.data, self.slices = torch.load(self.processed_paths[ConcatGeometryFile[geo_file].value]) # concat

    @property
    def raw_file_names(self):
        """ Same for all dataset types. """
        return ['/raw/train_reactants.sdf', '/raw/train_ts.sdf', '/raw/train_products.sdf', '/raw/test_reactants.sdf', '/raw/test_ts.sdf', '/raw/test_products.sdf']
    
    @property
    def processed_file_names(self):
        """ If files already in processed folder, this processing is skipped. 
            Convenient for accessing the individual processed files without having to recreate them each time. 
        """
        if self.dataset_type == "individual":
            return ['train_r.pt', 'train_ts.pt', 'train_p.pt', 'test_r.pt', 'test_ts.pt', 'test_p.pt']
        
        elif self.dataset_type == "concat":
            return ['train_concat_rp.pt', 'train_ts.pt', 'test_concat_rp.pt', 'test_ts.pt']
        
        else:
            raise ValueError('The dataset type you entered does not exist... try again with \'individual\' or \'concat\'.')

    def download(self):
        """ Not required in this project. """
        pass
    
    def process_geometry_file(self, geometry_file, list = None):
        """ Code mostly lifted from QM9 dataset creation https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html 
            Transforms molecules to their atom features and adjacency lists.
        """
        
        limit = TEMP_MOLS_LIMIT

        data_list = list if list else []
        full_path = self.root + geometry_file
        geometries = Chem.SDMolSupplier(full_path, removeHs=False, sanitize=False)

        # get atom and edge features for each geometry
        for i, mol in enumerate(tqdm(geometries)):

            # temp soln cos of split edge memory issues
            if i == limit:
                break
            
            N = mol.GetNumAtoms()
            # get atom positions as matrix w shape [num_nodes, num_dimensions] = [num_atoms, 3]
            atom_data = geometries.GetItemText(i).split('\n')[4:4 + N] 
            atom_positions = [[float(x) for x in line.split()[:3]] for line in atom_data]
            atom_positions = torch.tensor(atom_positions, dtype=torch.float)
            # all the features
            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []

            # atom/node features
            for atom in mol.GetAtoms():
                type_idx.append(self.types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridisation = atom.GetHybridization()
                sp.append(1 if hybridisation == HybridizationType.SP else 0)
                sp2.append(1 if hybridisation == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridisation == HybridizationType.SP3 else 0)
                # !!! should do the features that lucky does: whether bonded, 3d_rbf

            # bond/edge features
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds(): 
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                # edge type for each bond type; *2 because both ways
                edge_type += 2 * [self.bonds[bond.GetBondType()]]
            # edge_index is graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            # edge_attr is edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = F.one_hot(edge_type, num_classes=len(self.bonds)).to(torch.float) 

            # order edges based on combined ascending order
            perm = (edge_index[0] * N + edge_index[1]).argsort() # TODO
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            z = torch.tensor(atomic_number, dtype=torch.long)
            hs = (z == 1).to(torch.float) # hydrogens
            num_hs = scatter(hs[row], col, dim_size=N).tolist() # scatter helps with one-hot
            
            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float).t().contiguous()
            x = torch.cat([x1.to(torch.float), x2], dim=-1)

            data = Data(x=x, z=z, pos=atom_positions, edge_index=edge_index, edge_attr=edge_attr, idx=i)
            
            data_list.append(data)

        return data_list
        

    def process(self):
        """ Processes each of the six geometry files and appends to a list. """

        if self.dataset_type == "individual":
            
            for g_idx, geometry_file in enumerate(self.raw_file_names): 
                data_list = self.process_geometry_file(geometry_file)
                torch.save(self.collate(data_list), self.processed_paths[g_idx]) 
        
        elif self.dataset_type == "concat":

            # concat train r and p
            train_rp = []
            train_rp = self.process_geometry_file('/raw/train_reactants.sdf', train_rp)
            train_rp = self.process_geometry_file('/raw/train_products.sdf', train_rp) 
            torch.save(self.collate(train_rp), self.processed_paths[0])

            # train ts
            train_ts = self.process_geometry_file('/raw/train_ts.sdf')
            torch.save(self.collate(train_ts), self.processed_paths[1])

            # concat test r and p
            test_rp = []
            test_rp = self.process_geometry_file('/raw/test_reactants.sdf', test_rp)
            test_rp = self.process_geometry_file('/raw/test_products.sdf', test_rp) 
            torch.save(self.collate(test_rp), self.processed_paths[2])

            # test ts
            test_ts = self.process_geometry_file('/raw/test_ts.sdf')
            torch.save(self.collate(test_ts), self.processed_paths[3])


"""
    # TODO: implement transforms of atomic features to interatomic distances, etc.
    def coordinate_to_interatomic_dist():
        # maybe generalise this function with a flag for different initial inputs
            # e.g. interatomic distances, Z-matrix, nuclear charge, etc.
        # if these are different enough, may be better to have this instance as an abstract class then have implementations for each matrix type
    
    # TODO
    def visualise_feature_dynamics(self):
        # takes in reactants, ts, products; calculates reaction centre
        # compare how reaction centre changes: by how much, what precision do we need?
        # could do similar with interatomic distances
        return

    # not sure if necessary as can just iterate through
    def dataset_properties():
        # functions for rdkit molecule properties to compare
        return

    # other funcs: identifying 3D bias in dataset? perhaps need to take the original log files in to get a broader range of reactions
"""
