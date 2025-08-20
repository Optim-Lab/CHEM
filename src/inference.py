#%%
import wandb
import pandas as pd
import numpy as np
from copy import copy

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from torch_scatter import scatter_add

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
#%%
atom_dic = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
    19: 'K', 20: 'Ca'}

def visualize(graph, path=None, color=None, edge_color=None):
    if color==None:
        color=torch.arange(len(graph.x))
    G = to_networkx(graph, to_undirected=True)
    fig = plt.figure()
    nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G), alpha=0.7, node_size=200 ,with_labels=False,
                         node_color=color, edge_color=edge_color)
    
    label_list = [atom_dic.get(a.item(), 'X') for a in graph.x[:,0]]
    labels = {node: label for node, label in zip(G.nodes(), label_list)}
    nx.draw_networkx_labels(G, nx.kamada_kawai_layout(G), labels, font_size=12, font_color='black')
    
    if path is not None:
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path)
        try: wandb.log({'image': wandb.Image(path)})
        except: pass
        plt.close()
    else:
        # plt.show()
        plt.close()
    return fig

#%%
def mol_visualize(model, dataset, vis_num, path, args):
    if isinstance(vis_num, tuple):
        st, end = vis_num
        vis_num = range(st, end)
    model.eval()
    for j in vis_num:

        graph = Batch.from_data_list([dataset[j]])
        ncgc_id = dataset[j].id
        smiles = dataset[j].smiles
        original = copy(graph)
        att_score = model(graph.to(args.device), infer=True)
        _, _, pred, _ = model(graph.to(args.device))
        pred = torch.max(pred, dim=1)[1]
        
        if pred.item() == 1: path_ = path+'True/' 
        else: path_ = path+'False/' 

        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(mol, path_+f'{ncgc_id}_mol.png')
        color = ['blue' if item<0.5 else 'yellow' for item in att_score[:,1]]

        visualize(original, path_ + f'{ncgc_id}.png', color)

def causal_check(model, dataset, path, args):
    model.eval()
    binding =  pd.read_csv('../binding.csv')
    binding = binding.dropna(subset=['binding_node'])
    node_att = []
    label = []
    for row in binding.to_numpy():
        idx = row[2]
        graph = Batch.from_data_list([dataset[idx]])
        ncgc_id = dataset[idx].id
        assert ncgc_id == row[3]
        
        smiles = dataset[idx].smiles
        original = copy(graph)
        att_score = model(graph.to(args.device), infer=True)
        _, _, pred, _ = model(graph.to(args.device))
        pred = torch.max(pred, dim=1)[1].item()
        
        # mol = Chem.MolFromSmiles(smiles)
        # Draw.MolToFile(mol, path+f'{ncgc_id}_mol.png')
        
        true_cause = row[4]
        true_cause = set(map(int, true_cause.split(',')))
        color = []
        for i, node in enumerate(att_score[:,1]):
            node_att.append(node.item())
            label.append((i in true_cause))
            if (node >= 0.5)&(i in true_cause):
                color.append('red')
            elif (node >= 0.5)&(i not in true_cause):
                color.append('red')
            elif (node < 0.5)&(i in true_cause):
                color.append('gray')
            elif (node < 0.5)&(i not in true_cause):
                color.append('gray')

        visualize(original, path + f'{ncgc_id}_{pred}.pdf', color)
        # true_color = ['red' if i in true_cause else 'gray' for i in range(graph.num_nodes)]
        # visualize(original, path + f'{ncgc_id}_true.pdf', true_color)
    
    roc_auc = roc_auc_score(label, node_att)
    precision, recall, _ = precision_recall_curve(label, node_att)
    pr_auc = auc(recall, precision)
    
    return roc_auc, pr_auc

# %%
def sparsity(model, dataset, args, path=None):
    data_loader = DataLoader(dataset, 1, shuffle=False)
    model.eval()
    s = []
    att = []
    with torch.no_grad():
        for it, data in enumerate(data_loader):
            data = data.to(args.device)
            att_score = model(data, infer=True)
            att_node = scatter_add((att_score[:,1]>0.5).view(-1).to(int), data.batch)
            num_node = data.batch.bincount()
            s.extend((1-att_node/num_node).tolist())
            att.extend(att_score[:,1].tolist())
    if path is not None:
        plt.hist(att, density=True, color='#999999')
        plt.xlabel('Importance score', fontsize=20)
        plt.xticks([0,0.5,1], fontsize=20)
        plt.yticks(fontsize=20)
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        plt.tight_layout()
        plt.savefig(f'{path}hist_{args.seed}.pdf')
        plt.close()
    return np.mean(s), att
# %%