# pytorch, pyscatter, pyg
from numpy.lib.polynomial import polysub
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import InMemoryDataset, Data, DataLoader

# rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT

# other
from tqdm import tqdm

class OtherReactionTriple(Data):
    # seeing if this works

    def __init__(self, r = None, ts = None, p = None):
        super(OtherReactionTriple, self).__init__()

        # initial checks
        if r and ts and p:
            assert r.idx == ts.idx == p.idx, \
                "The IDs of each mol don't match. Are you sure your data processing is correct?"
            assert len(r.z) == len(ts.z) == len(p.z), \
                "The mols have different number of atoms."
            assert r.x.size(1) == ts.x.size(1) == p.x.size(1), \
                "You don't have the same number of atom features for each mol."
            assert r.edge_attr.size(1) == ts.edge_attr.size(1) == p.edge_attr.size(1), \
                "You don't have the same number of bond features for each mol."
            self.idx = r.idx
            self.num_atoms = len(r.z)
            self.num_atom_fs = r.x.size(1)
            self.num_bond_fs = r.edge_attr.size(1)

            # reactant
            self.edge_attr_r = r.edge_attr
            self.edge_index_r = r.edge_index
            self.pos_r = r.pos
            self.x_r = r.x
            self.z_r = r.z

            # ts
            self.edge_attr_ts = ts.edge_attr
            self.edge_index_ts = ts.edge_index
            self.pos_ts = ts.pos
            self.x_ts = ts.x
            self.z_ts = ts.z

            # product
            self.edge_attr_p = p.edge_attr
            self.edge_index_p = p.edge_index
            self.pos_p = p.pos
            self.x_p = p.x
            self.z_p = p.z
        else:
            NameError("Reactant, TS, or Product not defined for this reaction.")

    def __inc__(self, key, value):
        if key == 'edge_index_r' or key == 'edge_attr_r':
            return self.x_r.size(0)
        if key == 'edge_index_ts' or key == 'edge_attr_ts':
            return self.x_ts.size(0)
        if key == 'edge_index_p' or key == 'edge_attr_p':
            return self.x_p.size(0)
        else:
            return super().__inc__(key, value)
    
    def __cat_dim__(self, key, item):
        # NOTE: automatically figures out .x and .pos
        if key == 'edge_attr_r' or key == 'edge_attr_ts' or key == 'edge_attr_p':
            return 0
        if key == 'edge_index_r' or key == 'edge_index_ts' or key == 'edge_index_p':
            return 1
        else:
            return super().__cat_dim__(key, item)

class ReactionTriple(Data):
    """ Contains graph representation (as PyTorch Geometric Data() class) for reactant, TS, and product in a reaction trajectory.
    Notes:
        - Any params passed in must be set to default = None so that PyTorch's collate function works correctly. This collate fn
          helps us batch different graphs, too.
        - Key names are the variable names for params passed into __init__(). These are the same keys used for batching.
        - Different number of atoms could be strange when batching.
    TODO:
        - Functions for reaction core and data vis: rxn centre change magnitude (for standard/interatomic/etc.) -> precision required.
        - Atomic features to radial/interatomic/z-matrix, etc. Abstract class may be better.
        - Move number of atoms (and other checks?) into their own function.
        - Dataset properties: functions for rdkit molecule properties to compare
        - Identifying 3D bias in dataset?
    """

    def __init__(self, r = None, ts = None, p = None):
        super(ReactionTriple, self).__init__()
        
        # all molecules
        self.r = r
        self.ts = ts
        self.p = p

        if r and ts and p:
            assert len(r.z) == len(ts.z) == len(p.z)
            self.num_atoms = len(r.z)
        else:
            NameError("Reactant, TS, or Product not defined for this reaction.")

    def __inc__(self, key, value):
        """ Defines incremental count between two consecutive graph attributes.
        Notes:
            - Helps with batching individual parts of the reaction.
        
        Parameters
        ----------
        key : str 
            Either r, p, ts. Use this in follow batch to learn on this set of the trajectory.
        """
        if key == 'r':
            return self.r.x.size(0)
            # return self.r.edge_index.size(0)
        elif key == 'ts':
            return self.ts.x.size(0)
        elif key == 'p':
            return self.p.x.size(0)
        else:
            return super().__inc__(key, value)

class ReactionDataset(InMemoryDataset):
    """ Creates instance of reaction dataset, essentially a list of ReactionTriple(Reactant, TS, Product).
    TODO:
        - get_tracker(): factory method for wandb tracker log [ref pyg 3D]
    """
    TYPES = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    def __init__(self, root_folder, n_rxns, transform = None, pre_transform = None):
        self.n_rxns = n_rxns
        self.rxn_data = None
        super(ReactionDataset, self).__init__(root_folder, transform, pre_transform)

        # keep individual geometries stored in case
        self.r_data, self.r_slices = torch.load(self.processed_paths[0])
        self.ts_data, self.ts_slices = torch.load(self.processed_paths[1]) 
        self.p_data, self.p_slices = torch.load(self.processed_paths[2])
        
        # key data
        self.data, self.slices = torch.load(self.processed_paths[3])
        
    @property
    def raw_file_names(self):
        return ['/raw/train_reactants.sdf', '/raw/train_ts.sdf', '/raw/train_products.sdf', 
                        '/raw/test_reactants.sdf', '/raw/test_ts.sdf', '/raw/test_products.sdf']

    @property
    def processed_file_names(self):
        """ If files already in processed folder, this func means we don't have to recreate them. """
        return ['reactants.pt', 'ts.pt', 'products.pt', 'full.pt']
    
    def download(self):
        """ Not required in this project. """
        pass

    def process(self):
        """ Process reactants, TSs, and products. Store in their own files as well as together as reactions. 
        Notes:
            - Because of initial format of data (given as train and test), we pass the list of geometries forward.
        """

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
            # rxn = ReactionTriple(reactants[rxn_id], tss[rxn_id], products[rxn_id])
            rxn = OtherReactionTriple(reactants[rxn_id], tss[rxn_id], products[rxn_id])
            rxns.append(rxn)
        self.rxn_data = rxns # spare rxn_data if funny business with main rxns

        torch.save(self.collate(reactants), self.processed_paths[0])
        torch.save(self.collate(tss), self.processed_paths[1])
        torch.save(self.collate(products), self.processed_paths[2])
        torch.save(self.collate(rxns), self.processed_paths[3])

    def process_geometry_file(self, geometry_file, current_list = None):
        """ Transforms molecules to their atom features and adjacency lists.
        Notes:
            - Code mostly lifted from PyG QM9 dataset creation https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html 
        """
        data_list = current_list if current_list else []
        counted = len(data_list)
        full_path = self.root + geometry_file
        geometries = Chem.SDMolSupplier(full_path, removeHs = False, sanitize = False)

        # get atom and edge features for each geometry
        for i, mol in enumerate(tqdm(geometries)):

            if i == self.n_rxns:
                break

            N = mol.GetNumAtoms()
            
            #atom_positions = []
            #for c in mol.GetConformers():
            #    atom_positions.append(c.GetPositions())
            #atom_positions = torch.tensor(atom_positions, dtype = torch.float)
            
            # get atom positions as matrix w shape [num_nodes, num_dimensions] = [num_atoms, 3]
            atom_data = geometries.GetItemText(i).split('\n')[4: 4+N]
            atom_positions = [[float(x) for x in line.split()[:3]] for line in atom_data]
            atom_positions = torch.tensor(atom_positions, dtype = torch.float)
            
            # redoing positions
            #D = Chem.Get3DDistanceMatrix(mol)
            #atom_positions = []
            #for i in range(N):
            #    for j in range(i+1, N):
            #        atom_positions.extend([D[i][j], D[j][i]])
            #atom_positions = torch.tensor(atom_positions, dtype = torch.float)
            
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
            mol_data = Data(x = x, z = z, pos = atom_positions, edge_index = edge_index, edge_attr = edge_attr, idx = idx)
            data_list.append(mol_data)
        
        return data_list


        



    










