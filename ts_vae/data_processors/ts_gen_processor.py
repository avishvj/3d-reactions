import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from rdkit import Chem
from tqdm import tqdm

class TSGenGraph(Data):
    # seeing if this works

    def __init__(self, x = None, z = None, pos = None, edge_attr = None, idx = None):
        super(TSGenGraph, self).__init__(x = x, edge_attr = edge_attr, y = z, pos = pos)
        self.idx = idx

    def __inc__(self, key, value):
        if key == 'edge_attr' or key == 'edge_index':
            return self.x.size(0)
        else:
            return super().__inc__(key, value)
    
    def __cat_dim__(self, key, item):
        # NOTE: automatically figures out .x and .pos
        if key == 'edge_attr':
            return 0
        elif key == 'edge_index':
            return 1
        else:
            return super().__cat_dim__(key, item) 

class TSGenDataset(InMemoryDataset):
    """Creates instance of reaction dataset, essentially a list of ReactionTriple(Reactant, TS, Product)."""

    # constants
    MAX_D = 10.
    COORD_DIM = 3
    ELEM_TYPES = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
#    TEMP_MOLS_LIMIT = 10

    def __init__(self, root_folder, transform = None, pre_transform = None):
        super(TSGenDataset, self).__init__(root_folder, transform, pre_transform)
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
        r_train = Chem.SDMolSupplier(self.root + '/raw/train_reactants.sdf', removeHs = False, sanitize = False)
        r_test = Chem.SDMolSupplier(self.root + '/raw/test_reactants.sdf', removeHs = False, sanitize = False)
        rs = []
        for mol in r_train:
            rs.append(mol)
        for mol in r_test:
            rs.append(mol)
        
        # transition states
        ts_train = Chem.SDMolSupplier(self.root + '/raw/train_ts.sdf', removeHs = False, sanitize = False)
        ts_test = Chem.SDMolSupplier(self.root + '/raw/test_ts.sdf', removeHs = False, sanitize = False)
        tss = []
        for mol in ts_train:
            tss.append(mol)
        for mol in ts_test:
            tss.append(mol)
        
        # products
        p_train = Chem.SDMolSupplier(self.root + '/raw/train_products.sdf', removeHs = False, sanitize = False)
        p_test = Chem.SDMolSupplier(self.root + '/raw/test_products.sdf', removeHs = False, sanitize = False)
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

#            if rxn_id == self.TEMP_MOLS_LIMIT:
#                break

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
            
            node_feats = torch.tensor([type_ids, atomic_ns], dtype = torch.float).t().contiguous()
            atomic_ns = torch.tensor(atomic_ns, dtype = torch.long)
            edge_attr = torch.tensor([bonded, aromatic, rbf], dtype = torch.long).t().contiguous()

            data = TSGenGraph(x = node_feats, z = atomic_ns, pos = ts_gt_pos, edge_attr = edge_attr, idx = rxn_id)
            data_list.append(data) 

        return data_list

        



    









