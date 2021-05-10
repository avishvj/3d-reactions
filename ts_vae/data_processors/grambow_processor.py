import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm

class ReactionDataset(InMemoryDataset):
    """ Processes .sdf geometry files. """

    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    def __init__(self, root, transform=None, pre_transform=None):
        super(ReactionDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['/raw/train_reactants.sdf', '/raw/train_ts.sdf', '/raw/train_products.sdf', '/raw/test_reactants.sdf', '/raw/test_ts.sdf', '/raw/test_products.sdf']
    
    @property
    def processed_file_names(self):
        """ If files already in processed folder, this processing is skipped. """
        return ['train_r.pt', 'train_ts.pt', 'train_p.pt', 'test_r.pt', 'test_ts.pt', 'test_p.pt']

    def download(self):
        """ Not required in this project. """
        pass

    def process(self):
        """ Processes each of the six geometry files and appends to a list. 
            Code mostly lifted from QM9 dataset creation https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html 
        """

        for g_idx, geometry_file in enumerate(self.raw_file_names): # should maybe create enum with raw-processed together
            
            data_list = []
            full_path = self.root + geometry_file
            geometries = Chem.SDMolSupplier(full_path, removeHs=False, sanitize=False)
            
            for i, mol in enumerate(tqdm(geometries)):
                
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
                # one hot the edge types into distinct types for bonds
                # edge_attr is edge feature matrix with shape [num_edges, num_edge_features]
                edge_attr = F.one_hot(edge_type, num_classes=len(self.bonds)).to(torch.float) 

                # order edges based on combined ascending order
                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_type = edge_type[perm]
                edge_attr = edge_attr[perm]

                row, col = edge_index
                z = torch.tensor(atomic_number, dtype=torch.long)
                hs = (z == 1).to(torch.float) # hydrogens
                # https://abderhasan.medium.com/pytorchs-scatter-function-a-visual-explanation-351d25c05c73
                # helps with one-hot encoding, should come back to this
                num_hs = scatter(hs[row], col, dim_size=N).tolist() 
                
                x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types))
                x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float).t().contiguous()
                x = torch.cat([x1.to(torch.float), x2], dim=-1)

                # no direct y since plan to decode to TS
                data = Data(x=x, z=z, pos=atom_positions, edge_index=edge_index, edge_attr=edge_attr, idx=i)

                data_list.append(data)

                # if self.pre_filter is not None and not self.pre_filter(data):
                #     continue
                # if self.pre_transform is not None:
                #     data = self.pre_transform(data)

            torch.save(self.collate(data_list), self.processed_paths[g_idx]) 