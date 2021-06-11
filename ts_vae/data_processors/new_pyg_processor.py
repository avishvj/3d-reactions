# functions needed:
#   - collate(): collate data into batch format
#       - collate_fn specifies how exactly samples need to be batched
#       - collate_fn receives your __getitem__ func

# why use PyG... because of message passing? need to modify training method - that is the issue.


# pytorch, pyscatter, pyg
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

class ReactionTriple(Data):
    # TODO: funcs for reaction core, data vis

    def __init__(self, r = None, ts = None, p = None):
        # set defaults to None so that can work with collate function !! v important

        # TODO: make the key names clear here
        
        super(ReactionTriple, self).__init__()
        
        # all molecules
        self.r = r
        self.ts = ts
        self.p = p

        # number of atoms should be same, helps with batching, TODO: other checks?
        # TODO: maybe move this into its own func
        if r and ts and p:
            assert len(r.z) == len(ts.z) == len(p.z)
            self.num_atoms = len(r.z)

    def __inc__(self, key, value):
        # Defines incremental count between two consecutive graph attributes.
        if key == 'r':
            return self.r.edge_index.size(0)
        elif key == 'ts':
            return self.ts.edge_index.size(0)
        elif key == 'p':
            return self.p.edge_index.size(0)
        else:
            return super().__inc__(key, value)

class ReactionDataset(InMemoryDataset):
    # contains triples of reactant, ts, product
    # PyG could be better but run into issues with collate func:
    #   - why not just collate on batches (can do R, TS, P simultaneously because graphs based on max num_atoms in any of 3)
    # TODO: have get_tracker: factory method for wandb tracker log [ref pyg 3d]

    TYPES = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    TEMP_MOLS_LIMIT = 30

    def __init__(self, root_folder, transform = None, pre_transform = None):

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
        return ['reactants.pt', 'ts.pt', 'products.pt', 'full.pt']

    # my collate func to use instead of the PyG func
    # also how to use for my batches
    def collate_reactions(self):


        # keys = [Data params e.g. edge, etc.] data_list[0].keys = [r, p, ts]
        # class = Data()

        pass

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
        self.rxn_data = rxns

        torch.save(self.collate(reactants), self.processed_paths[0])
        torch.save(self.collate(tss), self.processed_paths[1])
        torch.save(self.collate(products), self.processed_paths[2])
        torch.save(self.collate(rxns), self.processed_paths[3])
        # TODO: modify collate for rxn triplets list

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
            mol_data = Data(x = x, z = z, pos = atom_positions, edge_index = edge_index, edge_attr = edge_attr, idx = idx)
            data_list.append(mol_data)
        
        return data_list


        



    










