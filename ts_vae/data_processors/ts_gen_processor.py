import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from rdkit import Chem
from tqdm import tqdm

class ReactionTriple(Data):
    # seeing if this works

    def __init__(self, r = None, ts = None, p = None):
        super(ReactionTriple, self).__init__()

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

            # ts
            self.edge_attr_ts = ts.edge_attr
            self.edge_index_ts = ts.edge_index
            self.pos_ts = ts.pos
            self.x_ts = ts.x

            # product
            self.edge_attr_p = p.edge_attr
            self.edge_index_p = p.edge_index
            self.pos_p = p.pos
            self.x_p = p.x
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

class ReactionDataset(InMemoryDataset):
    """Creates instance of reaction dataset, essentially a list of ReactionTriple(Reactant, TS, Product)."""

    # constants
    MAX_D = 10.
    COORD_DIM = 3
    ELEM_TYPES = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    TEMP_MOLS_LIMIT = 8000

    def __init__(self, root_folder, transform = None, pre_transform = None):
        super(ReactionDataset, self).__init__(root_folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['/raw/train_reactants.sdf', '/raw/train_ts.sdf', '/raw/train_products.sdf', 
                '/raw/test_reactants.sdf', '/raw/test_ts.sdf', '/raw/test_products.sdf']

    @property
    def processed_file_names(self):
        """If file already in processed folder, this func means we don't have to recreate them."""
        return ['full.pt']
    
    def download(self):
        """Not required in this project."""
        pass

    def process(self):
        """Process reactants, TSs, and products as reactions list."""

        # reactants
        r_train = Chem.SDMolSupplier('data/raw/train_reactants.sdf', removeHs = False, sanitize = False)
        r_test = Chem.SDMolSupplier('data/raw/test_reactants.sdf', removeHs = False, sanitize = False)
        rs = []
        for mol in r_train:
            rs.append(mol)
        for mol in r_test:
            rs.append(mol)
        
        # transition states
        ts_train = Chem.SDMolSupplier('data/raw/train_ts.sdf', removeHs = False, sanitize = False)
        ts_test = Chem.SDMolSupplier('data/raw/test_ts.sdf', removeHs = False, sanitize = False)
        tss = []
        for mol in ts_train:
            tss.append(mol)
        for mol in ts_test:
            tss.append(mol)
        
        # products
        p_train = Chem.SDMolSupplier('data/raw/train_products.sdf', removeHs = False, sanitize = False)
        p_test = Chem.SDMolSupplier('data/raw/test_products.sdf', removeHs = False, sanitize = False)
        ps = []
        for mol in p_train:
            ps.append(mol)
        for mol in p_test:
            ps.append(mol)
        
        assert len(rs) == len(tss) == len(ps), f"Lengths of reactants ({len(rs)}), transition states \
                                                ({len(tss)}), products ({len(ps)}) don't match."

        geometries = list(zip(rs, tss, ps))
        data_list = self.process_geometries(geometries)
        torch.save(self.collate(data_list), self.processed_paths[0])

    def process_geometries(self, geometries):
        """Process all geometries in same manner as ts_gen."""
        
        data_list = []
        
        for rxn_id, rxn in enumerate(tqdm(geometries)):
            r, ts, p = rxn
            num_atoms = r.GetNumAtoms()

            # dist matrices
            D = (Chem.GetDistanceMatrix(r) + Chem.GetDistanceMatrix(p)) / 2
            D[D > self.MAX_D] = self.MAX_D
            D_3D_rbf = np.exp(-((Chem.Get3DDistanceMatrix(r) + Chem.Get3DDistanceMatrix(p)) / 2))  

            # node feats, edge attr init
            type_ids, atomic_ns = [], []
            bonded, aromatic, rbf = [], [], []
            
            # ts ground truth coords
            ts_gt_pos = torch.zeros((num_atoms, self.COORD_DIM))
            ts_conf = ts.GetConformer()

            for i in range(num_atoms):

                # node feats
                atom = r.GetAtomWithIdx(i)
                type_ids.append(self.ELEM_TYPES[atom.GetSymbol()])
                atomic_ns.append(atom.GetAtomicNum() / 10.)

                # ts coordinates: atom positions as matrix w shape [num_atoms, 3]
                pos = ts_conf.GetAtomPosition(i)
                ts_gt_pos[i] = torch.tensor([pos.x, pos.y, pos.z])
                
                # edge attrs
                for j in range(num_atoms):
                    # if stays bonded
                    if D[i][j] == 1: 
                        bonded.append(1)
                        aromatic.append(1 if r.GetBondBetweenAtoms(i, j).GetIsAromatic() else 0)
                    else:
                        bonded.append(0)
                        aromatic.append(0)
                    # 3d rbf
                    rbf.append(D_3D_rbf[i][j])
            
            node_feats = torch.tensor([type_ids, atomic_ns], dtype = torch.float)
            atomic_ns = torch.tensor(atomic_ns, dtype = torch.long)
            edge_attr = torch.tensor([bonded, aromatic, rbf], dtype = torch.long)

            data = Data(x = node_feats, z = atomic_ns, pos = ts_gt_pos, edge_attr = edge_attr, idx = rxn_id)
            data_list.append(data) 

        return data_list

        



    










