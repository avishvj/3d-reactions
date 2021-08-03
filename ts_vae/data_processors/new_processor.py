import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from tqdm import tqdm








# make sure same as other workspace












class TSGenDataset(InMemoryDataset):
    """Creates instance of reaction dataset, essentially a list of ReactionTriple(Reactant, TS, Product)."""
    MAX_D = 10.
    COORD_DIM = 3
    ELEM_TYPES = {'H': 0, 'C': 1, 'N': 2, 'O': 3}
    NUM_ELEMS = len(ELEM_TYPES)
    TEMP_MOLS_LIMIT = 40
    NUM_EDGE_ATTR = 3
    NUM_NODE_FEATS = 5
    MAX_NUM_ATOMS = 21

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
        
        sanitise = False

        # reactants
        r_train = Chem.SDMolSupplier(self.root + '/raw/train_reactants.sdf', removeHs = False, sanitize = sanitise)
        r_test = Chem.SDMolSupplier(self.root + '/raw/test_reactants.sdf', removeHs = False, sanitize = sanitise)
        rs = []
        for mol in r_train:
            rs.append(mol)
        for mol in r_test:
            rs.append(mol)
        
        # transition states
        ts_train = Chem.SDMolSupplier(self.root + '/raw/train_ts.sdf', removeHs = False, sanitize = sanitise)
        ts_test = Chem.SDMolSupplier(self.root + '/raw/test_ts.sdf', removeHs = False, sanitize = sanitise)
        tss = []
        for mol in ts_train:
            tss.append(mol)
        for mol in ts_test:
            tss.append(mol)
        
        # products
        p_train = Chem.SDMolSupplier(self.root + '/raw/train_products.sdf', removeHs = False, sanitize = sanitise)
        p_test = Chem.SDMolSupplier(self.root + '/raw/test_products.sdf', removeHs = False, sanitize = sanitise)
        ps = []
        for mol in p_train:
            ps.append(mol)
        for mol in p_test:
            ps.append(mol)
        
        assert len(rs) == len(tss) == len(ps), f"Lengths of reactants ({len(rs)}), transition states \
                                                ({len(tss)}), products ({len(ps)}) don't match."

        data = list(zip(rs, tss, ps))
        data_list = self.process_geometries(data)
        torch.save(self.collate(data_list), self.processed_paths[0])
    
    def process_geometries(self, geometries):

        data_list = []
        
        for rxn_id, rxn in enumerate(tqdm(geometries)):
            
            if rxn_id == self.TEMP_MOLS_LIMIT:
                break

            r, ts, p = rxn
            
            num_atoms = r.GetNumAtoms()
            
            f_bonds, edge_index, y = [], [], []
            f_atoms = torch.zeros(self.MAX_NUM_ATOMS, self.NUM_NODE_FEATS, dtype = torch.float)

            # topological and 3d distance matrices, NOTE: topological currently unused
            tD_r = Chem.GetDistanceMatrix(r)
            tD_p = Chem.GetDistanceMatrix(p)
            D_r = Chem.Get3DDistanceMatrix(r)
            D_p = Chem.Get3DDistanceMatrix(p)
            D_ts = Chem.Get3DDistanceMatrix(ts)

            for i in range(num_atoms):

                # node feats
                atom = r.GetAtomWithIdx(i)
                e_ix = self.ELEM_TYPES[atom.GetSymbol()]
                f_atoms[i][e_ix] = 1
                f_atoms[i][self.NUM_ELEMS] = atom.GetAtomicNum() / 10.

                # edge features
                for j in range(i+1, num_atoms):
                    
                    # fully connected graph
                    edge_index.extend([(i, j), (j, i)])

                    # for now, naively include both reac and prod
                    b1_feats = [D_r[i][j], D_p[i][j]]
                    b2_feats = [D_r[j][i], D_p[j][i]]

                    f_bonds.append(b1_feats)
                    f_bonds.append(b2_feats)
                    y.extend([D_ts[i][j], D_ts[j][i]])

            node_feats = torch.tensor(f_atoms, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(f_bonds, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)

            data = Data(x = node_feats, edge_attr = edge_attr, edge_index = edge_index, y = y, idx = rxn_id)
            data_list.append(data) 
        
        return data_list