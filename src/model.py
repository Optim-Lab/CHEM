import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import GINConv, GATConv
from torch_geometric.data import Batch
from gcn_conv import GCNConv

from torch_scatter import scatter_mean
from ogb.utils.features import get_atom_feature_dims
import copy

full_atom_feature_dims = get_atom_feature_dims()

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim+1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class MLP_layer(nn.Module):
    def __init__(self, hidden, hidden_out):
        super().__init__()
        self.fc1_bn = BatchNorm1d(hidden)
        self.fc1 = Linear(hidden, hidden)
        self.fc2_bn = BatchNorm1d(hidden)
        self.fc2 = Linear(hidden, hidden_out)
        
    def forward(self, x):
        x = self.fc1_bn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2_bn(x)
        x = self.fc2(x)
        return x
    
class GNN_layer(nn.Module):
    def __init__(self, model, hidden, dropout, residual):
        super().__init__()    
        self.bn = BatchNorm1d(hidden)
        if model == 'GCN':
            self.conv = GCNConv(hidden, hidden)
        elif model == 'GIN':
            self.conv = GINConv(
                Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden)
                )
            )
        elif model == 'GAT':
            self.conv = GATConv(hidden, hidden, heads=4, concat=False)
            
        self.dropout = dropout
        self.residual = residual
        
    def forward(self, x, edge_index):
        x_ = x.clone()
        x = self.bn(x)
        x = self.conv(x, edge_index)
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        if self.residual:
            x = x_ + x
        return x

class CAL_GCN(nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                    args, 
                    dropout=0.5):
        super(CAL_GCN, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout

        self.without_edge_attention = args.without_edge_attention
        out_dim = args.num_classes*args.num_task
              
        self.atom_encoder = AtomEncoder(hidden)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(
                GNN_layer(args.model, hidden, dropout, args.residual))
                
        self.node_att_mlp = MLP_layer(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GCNConv(hidden, hidden)
        self.causal_convs = GCNConv(hidden, hidden)

        # causla mlp
        self.causal_mlp = MLP_layer(hidden, out_dim)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.random_mlp = MLP_layer(hidden*2, out_dim)    
        elif self.args.cat_or_add == "add":
            self.random_mlp = MLP_layer(hidden, out_dim) 
        else:
            assert False

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, infer=False, silence_node=None, noise_type='zero'):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch

        x = self.atom_encoder(x)    
        if (silence_node is not None)&(noise_type == 'rand'): 
            x[silence_node] = torch.randn_like(x[silence_node])
        elif (silence_node is not None)&(noise_type == 'zero'): 
            x[silence_node] = torch.zeros_like(x[silence_node])

        x, edge_index, is_clique, clique2node = self.motif_pooling(data, x, batch)
        
        for conv in self.convs:
            x = conv(x, edge_index)
        
        x_ = x[is_clique==1]
        motif_att = self.node_att_mlp(x_)
        node_att = motif_att[clique2node]
        
        if infer: return F.softmax(node_att, dim=-1)

        node_mask = F.gumbel_softmax(node_att, hard=False)
        node_mask_ = F.gumbel_softmax(node_att, hard=False)

        x = x[is_clique==0]
        edge_index = data.edge_index
        
        # edge attention           
        row, col = edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2).cuda()
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        epsilon = torch.mean(scatter_mean(x, batch, dim=0)[batch], dim=-1).view(-1,1)
        
        xs = node_mask[:, 0].view(-1, 1) * x + (1-node_mask[:, 0].view(-1, 1))*epsilon
        xc = node_mask_[:, 1].view(-1, 1) * x + (1-node_mask_[:, 1].view(-1, 1))*epsilon

        xs = F.relu(self.context_convs(self.bnc(xs), edge_index, edge_weight_c))
        xc = F.relu(self.causal_convs(self.bno(xc), edge_index, edge_weight_o))

        xs = self.global_pool(xs, batch)
        xc = self.global_pool(xc, batch)
        
        if silence_node is not None: return node_att, self.random_readout_layer(xs, xc)
        
        c_logit = self.causal_mlp(xc)
        cs_logit = self.random_readout_layer(xs, xc)
        
        if self.args.const: 
            sparsity_loss = torch.sum(F.softmax(motif_att, dim=-1)[:,1])/data.num_graphs
        else:
            sparsity_loss = 0

        return c_logit, cs_logit, sparsity_loss

    def random_readout_layer(self, xs, xc):
        num = xs.shape[0]
        if self.training:
            random_idx = torch.randperm(num)
        else:
            random_idx = torch.arange(num)
            
        if self.args.cat_or_add == "cat":
            x = torch.cat((xs[random_idx], xc), dim=1)
        else:
            x = xs[random_idx] + xc
        return self.random_mlp(x)
    
    def motif_pooling(self, data, x, batch):
        data_list = copy.deepcopy(data.to_data_list())
        offset = 0
        for idx, data_point in enumerate(data_list):
            clique = data_point.clique
            data_point.x = x[batch ==idx]
            
            if self.args.pool=='add': 
                pool_x = global_add_pool(data_point.x[sum(clique, [])], data_point.node2clique)
            elif self.args.pool=='mean':
                pool_x = global_mean_pool(data_point.x[sum(clique, [])], data_point.node2clique)
            elif self.args.pool=='max':
                pool_x = global_max_pool(data_point.x[sum(clique, [])], data_point.node2clique)
            elif self.args.pool=='att':
                pool_x = self.global_attention(data_point.x[sum(clique, [])], data_point.node2clique)
            else:
                raise Exception("pool option not valid")
            data_point.clique2node += offset
            offset += data_point.num_cliques.item()
            
            data_point.num_nodes += len(clique)
            data_point.edge_index = data_point.hi_edge_index
            data_point.x = torch.cat([data_point.x, pool_x], dim=0)
            
        new_data = Batch.from_data_list(data_list)
        return new_data.x, new_data.edge_index, new_data.is_clique, new_data.clique2node
