import tempfile
import pandas as pd
import itertools
import torch
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles

from ogb.graphproppred import PygGraphPropPredDataset

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from chemutils import brics_decomp, get_mol, tree_decomp

RDLogger.DisableLog('rdApp.*')

def get_random_split(length):
    split_idx = {}
    ratio = round(length*0.1)
    train_idx = torch.randperm(length)
    split_idx["test"] = train_idx[:ratio]
    split_idx["valid"] = train_idx[ratio:ratio*2]
    split_idx["train"] = train_idx[ratio*2:]
    return split_idx

# Convert mutag data to smiles and fix chemically incorrect parts
def get_MUTAG_smiles():
    node_label_map = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
    edge_label_map = {0: Chem.BondType.AROMATIC, 1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    dataset = TUDataset(root='./dataset', name='MUTAG')
    smiles_list = []
    for idx, data in enumerate(dataset):

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_labels = data.x.squeeze().tolist()

        if (idx in [82,187]): edge_attr[22] = torch.tensor([0.,1.,0.,0.])

        mol = Chem.RWMol()
        
        for node_label in node_labels:
            atom = Chem.Atom(node_label_map[node_label.index(1.0)])
            mol.AddAtom(atom)

        for k, (i, j) in enumerate(edge_index.t().tolist()):
            bond_type = edge_label_map[torch.argmax(edge_attr[k]).item()]
            try: mol.AddBond(int(i), int(j), bond_type)
            except: pass
            atom_i = mol.GetAtomWithIdx(int(i))
            atom_j = mol.GetAtomWithIdx(int(j))
            
            if bond_type == Chem.BondType.SINGLE:
                if (atom_i.GetSymbol() == 'N' and atom_j.GetSymbol() == 'O'):
                    atom_i.SetFormalCharge(1)
                    atom_j.SetFormalCharge(-1)
                elif (atom_i.GetSymbol() == 'O' and atom_j.GetSymbol() == 'N'):
                    atom_i.SetFormalCharge(-1)
                    atom_j.SetFormalCharge(1)
        
        if (idx in [13,41,88,119,137,177]): mol.GetAtomWithIdx(1).SetFormalCharge(1)
        if (idx==149): mol.GetAtomWithIdx(5).SetFormalCharge(1)
                    
        AllChem.SanitizeMol(mol)  
        smiles = Chem.MolToSmiles(mol)
        smiles_list.append(smiles)
    with open('./dataset/MUTAG/raw/MUTAG_smiles.txt', '+w') as f:
        f.write('\n'.join(smiles_list))

def get_MUTAG():
    dataset = TUDataset(root='./dataset', name='MUTAG')
    smiles_list = []
    try:
        with open('./dataset/MUTAG/raw/MUTAG_smiles.txt', 'r') as f:
            for line in f:
                smiles_list.append(line.strip())
    except:
        get_MUTAG_smiles()
        with open('./dataset/MUTAG/raw/MUTAG_smiles.txt', 'r') as f:
            for line in f:
                smiles_list.append(line.strip())

    datalist = []
    for idx, data in enumerate(dataset):
        mol = get_mol(smiles_list[idx])
        y = data.y.squeeze()
        data = from_smiles(smiles_list[idx])
        data.y = y
        
        if mol is None: continue
        clique, edges = brics_decomp(mol)
        if len(edges)==0:
            clique, edges = tree_decomp(mol)

        pool_edge_index = torch.tensor([])
        if len(edges)!=0:
            pool_edge_index = torch.tensor(edges).mT
            pool_edge_index = torch.cat([pool_edge_index, pool_edge_index[[1,0]]], dim=-1)
            
        data.clique = clique
        data.pool_edge_index = pool_edge_index.to(torch.long)
        data.is_clique = torch.cat([torch.zeros(data.num_nodes), 
                                    torch.ones(len(clique))]).to(torch.long)
        
        node2clique = []
        clique2node = torch.zeros(data.num_nodes, dtype=torch.long)
        for i, sublist in enumerate(clique):
            node2clique.extend([i] * len(sublist))
            clique2node[sublist] = i
        data.node2clique = torch.tensor(node2clique, dtype=torch.long)
        data.clique2node = clique2node
        
        hi_edge = torch.tensor([list(itertools.chain(*clique)),data.node2clique+data.num_nodes])
        hi_edge = torch.cat([hi_edge, hi_edge[[1,0]]], dim=-1)
        
        data.hi_edge_index = torch.cat(
            [data.edge_index, data.pool_edge_index+data.num_nodes, hi_edge],
            dim=-1)
        
        data.num_cliques = len(clique)
        
        datalist.append(data)
          
    return MoleculeDataset('./', datalist), get_random_split(len(datalist))

def get_MolculeNetData(name, target_col=None):
    name = name.lower()
    dataset = PygGraphPropPredDataset(name='ogbg-mol'+name)
    smiles = pd.read_csv(f'./dataset/ogbg_mol{name}/mapping/mol.csv.gz', compression = 'gzip')
    smiles = smiles['smiles'].to_list()

    datalist = []
    for idx, data in enumerate(dataset):
        data.smiles = smiles[idx]
        if target_col is None:
            data.y = data.y.squeeze()
        else:
            data.y = data.y.squeeze()[target_col]
        
        mol = get_mol(data.smiles)
        if mol is None: continue
        clique, edges = brics_decomp(mol)
        if len(edges)==0:
            clique, edges = tree_decomp(mol)

        pool_edge_index = torch.tensor([])
        if len(edges)!=0:
            pool_edge_index = torch.tensor(edges).mT
            pool_edge_index = torch.cat([pool_edge_index, pool_edge_index[[1,0]]], dim=-1)
            
        data.clique = clique
        data.pool_edge_index = pool_edge_index.to(torch.long)
        data.is_clique = torch.cat([torch.zeros(data.num_nodes), 
                                    torch.ones(len(clique))]).to(torch.long)
        
        node2clique = []
        clique2node = torch.zeros(data.num_nodes, dtype=torch.long)
        for i, sublist in enumerate(clique):
            node2clique.extend([i] * len(sublist))
            clique2node[sublist] = i
        data.node2clique = torch.tensor(node2clique, dtype=torch.long)
        data.clique2node = clique2node
        
        hi_edge = torch.tensor([list(itertools.chain(*clique)),data.node2clique+data.num_nodes])
        hi_edge = torch.cat([hi_edge, hi_edge[[1,0]]], dim=-1)
        
        data.hi_edge_index = torch.cat(
            [data.edge_index, data.pool_edge_index+data.num_nodes, hi_edge],
            dim=-1)
        
        data.num_cliques = len(clique)
        
        datalist.append(data)
   
    return MoleculeDataset('./', datalist), dataset.get_idx_split()


def get_Tox21Data():
    dataset = pd.read_csv('./dataset/tox21_v2.csv')
    dataset = dataset.iloc[:,[5, 12,13]].dropna()
    dataset = dataset.to_numpy()
    
    datalist = []
    for datapoint in dataset:
        data = from_smiles(datapoint[-1])
        if data.x.size()[0] == 0: continue
        if len(datapoint) == 3:
            data.y = torch.tensor(datapoint[0], dtype=torch.long)
        else:
            data.y = torch.tensor(datapoint[:-2].astype(float))
        data.id = datapoint[-2]
        
        mol = get_mol(data.smiles)
        if mol is None: continue
        clique, edges = brics_decomp(mol)
        if len(edges)==0:
            clique, edges = tree_decomp(mol)

        pool_edge_index = torch.tensor([])
        if len(edges)!=0:
            pool_edge_index = torch.tensor(edges).mT
            pool_edge_index = torch.cat([pool_edge_index, pool_edge_index[[1,0]]], dim=-1)
            
        data.clique = clique
        data.pool_edge_index = pool_edge_index.to(torch.long)
        data.is_clique = torch.cat([torch.zeros(data.num_nodes), 
                                    torch.ones(len(clique))]).to(torch.long)
        
        node2clique = []
        clique2node = torch.zeros(data.num_nodes, dtype=torch.long)
        for i, sublist in enumerate(clique):
            node2clique.extend([i] * len(sublist))
            clique2node[sublist] = i
        data.node2clique = torch.tensor(node2clique, dtype=torch.long)
        data.clique2node = clique2node
        
        hi_edge = torch.tensor([list(itertools.chain(*clique)),data.node2clique+data.num_nodes])
        hi_edge = torch.cat([hi_edge, hi_edge[[1,0]]], dim=-1)
        
        data.hi_edge_index = torch.cat(
            [data.edge_index, data.pool_edge_index+data.num_nodes, hi_edge],
            dim=-1)
        
        data.num_cliques = len(clique)
        
        datalist.append(data)
        
    binding =  pd.read_csv('./dataset/binding.csv')
    binding = binding.dropna(subset=['binding_node'])
    idx = binding['idx'].tolist()
    
    split_idx = {}
    ratio = round(len(datalist)*0.1)
    train_idx = list(set(range(len(datalist))) - set(idx))
    random.shuffle(train_idx)
    train_idx = torch.tensor(train_idx)
    
    split_idx["test"] = torch.cat([torch.tensor(idx), train_idx[:ratio]])
    split_idx["valid"] = train_idx[ratio:2*ratio]
    split_idx["train"] = train_idx[2*ratio:]
   
    return MoleculeDataset('./', datalist), split_idx

def get_SynChem(bias):
    train_data = pd.read_csv(f'./dataset/syn_chem/train_{bias}.csv')
    val_data = pd.read_csv(f'./dataset/syn_chem/val_{bias}.csv')
    test_data = pd.read_csv(f'./dataset/syn_chem/test.csv')
    
    split_idx = {}
    split_idx["train"] = torch.arange(0,len(train_data))
    split_idx["valid"] = torch.arange(len(train_data),len(train_data)+len(val_data))
    split_idx["test"] = torch.arange(len(train_data)+len(val_data), len(train_data)+len(val_data)+len(test_data))
    
    dataset = pd.concat([train_data, val_data, test_data], axis=0)
    
    datalist = []
    for row in dataset.values:
        
        mol = get_mol(row[0])
        data = from_smiles(row[0])
        data.y = row[1]
        
        if mol is None: continue
        clique, edges = brics_decomp(mol)
        if len(edges)==0:
            clique, edges = tree_decomp(mol)

        pool_edge_index = torch.tensor([])
        if len(edges)!=0:
            pool_edge_index = torch.tensor(edges).mT
            pool_edge_index = torch.cat([pool_edge_index, pool_edge_index[[1,0]]], dim=-1)
            
        data.clique = clique
        data.pool_edge_index = pool_edge_index.to(torch.long)
        data.is_clique = torch.cat([torch.zeros(data.num_nodes), 
                                    torch.ones(len(clique))]).to(torch.long)
        
        node2clique = []
        clique2node = torch.zeros(data.num_nodes, dtype=torch.long)
        for i, sublist in enumerate(clique):
            node2clique.extend([i] * len(sublist))
            clique2node[sublist] = i
        data.node2clique = torch.tensor(node2clique, dtype=torch.long)
        data.clique2node = clique2node
        
        hi_edge = torch.tensor([list(itertools.chain(*clique)),data.node2clique+data.num_nodes])
        hi_edge = torch.cat([hi_edge, hi_edge[[1,0]]], dim=-1)
        
        data.hi_edge_index = torch.cat(
            [data.edge_index, data.pool_edge_index+data.num_nodes, hi_edge],
            dim=-1)
        
        data.num_cliques = len(clique)
        
        datalist.append(data)
          
    return MoleculeDataset('./', datalist), split_idx

# %%
class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        self._temp_dir = tempfile.TemporaryDirectory()
        super().__init__(self._temp_dir.name, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        self.save(self.data_list, self.processed_paths[0])
        
#%%