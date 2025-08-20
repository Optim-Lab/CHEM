#%%
from rdkit import Chem
import random
import os
from rdkit import RDLogger
import pandas as pd
RDLogger.DisableLog('rdApp.*')

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


chain_scaffolds = [
    'CCCCCCCCCCCCCC', # Phenanthrene
    'CCCCCCCCCCCCCCC',# Terphenyl
    'CCCCCCCCCC(CC)C', # p-Terphenyl
    'CCCCCCCCCCCC(C)C',   # Benzo[a]pyrene
    'CCCC(C)CCC(C)CCCCCCC', # Anthracene fused to phenyl
    'CCCC(C)CCC(C)CCC(C)C', # Benz[a]anthracene
    'CC(C)CCC(C)CCC(C)C', # Benz[a]anthracene
]

ring_scaffolds = [
    'C1=CC=C2C(=C1)C=CC3=CC=CC=C32', # Phenanthrene
    'C1=CC2=CC=CC=C2C=C1C3=CC=CC=C3',# Terphenyl
    'C1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3', # p-Terphenyl
    'C1=CC=C2C=C3C=CC=CC3=CC2=C1',   # Benzo[a]pyrene
    'C1=CC=C2C=CC=CC2=C1C3=CC=CC=C3', # Anthracene fused to phenyl
    'C1=CC=C(C=C1)C2=CC=CC3=CC=CC=C23', # Benz[a]anthracene
    'C1=CC2=C(C=C1)C3=CC=CC=C3C=C2',      # Chrysene
    'C1=CC=C2C(=C1)C3=CC=CC=C3C4=CC=CC=C24', # Tetracene
    'C1=CC2=C3C=CC=CC3=CC3=CC=CC=C3C2=C1', # Perylene
    'C1=CC2=CC=CC=C2C=C1C3=CC=CC=C3C4=CC=CC=C4', # Pentacene derivative
    'C1=CC2=C3C=CC=CC3=CC3=C4C=CC=CC4=CC=C3C2=C1',
    'C1=CC2=C3C=CC=CC3=CC3=CC=CC=C3C4=CC=CC=C4C2=C1',
    'C1=CC2=C(C=C1)C3=CC=CC=C3C4=CC=CC=C4C2=C5C=CC=CC5',
    'C1=CC2=C(C=C1)C3=CC=CC=C3C4=CC=CC=C4C5=CC=CC=C25',
    'C1=CC2=CC=CC=C2C=C1C3=CC=CC=C3C4=CC=CC=C4C5=CC=CC=C5',
    'C1=CC2=C(C=C1)C3=CC=CC=C3C2=C4C=CC=CC4'
]

np_substituents = {
'[O][CH3]': [1],  
'[O]CC': [0],  
'[Cl]': [0],
'C=CCCl': [0,1,2],
'[N](CC)CC': [0],
'C1CCCO1': [0,1,2,3], 
'[N]1CCCCC1': [0],
}

np_substituents2 = {
'[N](C)C': [0], 
'[Br]': [0],
'C(Cl)(Br)C': [0,3],
'[O]C(C)C': [0],
}

p_substituents = {
'[N](C)C': [1,2], 
'[N](CC)CC': [1,2,3,4],
'[CH2]c1cc[nH]c1': [0], 
'[O]C1CCCO1': [0], 
'[OH]C1CCCO1': [2,3], 
'[O]1CC1': [1,2],  
'[OH]': [0],
'CC(=O)Br': [0],
}

p_substituents2 = {
'[O]CC': [1,2],  
'[O]C(=O)C': [0,3],  
'CC(=O)Cl': [0],
'[NH2]': [0],
'C(=O)O': [0,2],
}

def attach_group(base_smiles, group_smiles, bond_idx):
    base_mol = Chem.MolFromSmiles(base_smiles)
    group_mol = Chem.MolFromSmiles(group_smiles)

    combo = Chem.CombineMols(base_mol, group_mol)
    rw_mol = Chem.RWMol(combo)

    base_atoms = base_mol.GetNumAtoms()
    group_anchor_idx = base_atoms + random.choice(bond_idx)

    carbon_indices = [atom.GetIdx() for atom in base_mol.GetAtoms() if atom.GetSymbol() == 'C']

    target_idx = random.choice(carbon_indices)

    rw_mol.AddBond(target_idx, group_anchor_idx, Chem.rdchem.BondType.SINGLE)

    try:
        Chem.SanitizeMol(rw_mol)
        return Chem.MolToSmiles(rw_mol)
    except:
        # print('Fail!!')
        return None
    
def make_graph(N, scaffolds, substituents):
    smiles_list = set()
    while(1):
        scaffold = random.choice(scaffolds)
        group = random.choice(list(substituents.keys()))
        bond_idx = substituents[group]
        smiles = attach_group(scaffold, group, bond_idx)
        if smiles == None: continue
        smiles_list.add(smiles)
        print(smiles)
        if len(smiles_list) == N: break
    return list(smiles_list)

# %%
def make_graph_dataset(n = 1000, bias = 0.9, seed=0):
    random.seed(seed)
    num_graph = round(n*18/40)

    cp = make_graph(num_graph, chain_scaffolds, p_substituents)
    cn = make_graph(num_graph, chain_scaffolds, np_substituents)
    rp = make_graph(num_graph, ring_scaffolds, p_substituents)
    rn = make_graph(num_graph, ring_scaffolds, np_substituents)

    cp = pd.DataFrame([(x,1,'cp') for x in cp], columns=['smiles', 'label', 'home']).iloc[:round(num_graph*bias)]
    cn = pd.DataFrame([(x,0,'cn') for x in cn], columns=['smiles', 'label', 'home']).iloc[:round(num_graph*(1-bias))]
    rp = pd.DataFrame([(x,1,'rp') for x in rp], columns=['smiles', 'label', 'home']).iloc[:round(num_graph*(1-bias))]
    rn = pd.DataFrame([(x,0,'rn') for x in rn], columns=['smiles', 'label', 'home']).iloc[:round(num_graph*bias)]

    train_val_df = pd.concat([cp, cn, rp, rn], axis=0).reset_index(drop=True)

    train_val_df = train_val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(len(train_val_df)/9)
    train_df = train_val_df.iloc[train_size:]
    val_df = train_val_df.iloc[:train_size]
    
    path = './dataset/syn_chem'
    if not os.path.exists(path):
        os.makedirs(path)
    train_df.to_csv(f'{path}/train_{bias}.csv', index=False)
    val_df.to_csv(f'{path}/val_{bias}.csv', index=False)
    
    return None
    
def make_graph_test_dataset(n = 1000, seed=0):
    random.seed(seed)
    cp = make_graph(round(n/40), chain_scaffolds, p_substituents2)
    cn = make_graph(round(n/40), chain_scaffolds, np_substituents2)
    rp = make_graph(round(n/40), ring_scaffolds, p_substituents2)
    rn = make_graph(round(n/40), ring_scaffolds, np_substituents2)

    cp = pd.DataFrame([(x,1,'cp') for x in cp], columns=['smiles', 'label', 'home'])
    cn = pd.DataFrame([(x,0,'cn') for x in cn], columns=['smiles', 'label', 'home'])
    rp = pd.DataFrame([(x,1,'rp') for x in rp], columns=['smiles', 'label', 'home'])
    rn = pd.DataFrame([(x,0,'rn') for x in rn], columns=['smiles', 'label', 'home'])
    
    test_df = pd.concat([cp, cn, rp, rn], axis=0).reset_index(drop=True)

    path = './dataset/syn_chem'
    if not os.path.exists(path):
        os.makedirs(path)
        
    test_df.to_csv(f'{path}/test.csv', index=False)
    return None
# %%
if __name__ == "__main__":
    make_graph_test_dataset(200)
    make_graph_dataset(n = 200, bias = 0.5)
    make_graph_dataset(n = 200, bias = 0.7)
    make_graph_dataset(n = 200, bias = 0.9)
# %%
