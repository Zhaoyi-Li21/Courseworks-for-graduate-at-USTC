import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch_geometric.nn import PairNorm

class GCN(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, dropout, 
                 layer_num=2, activation='relu', drop_edge=False, pair_norm=False):
        super(GCN, self).__init__()

        self.gc_inp = GraphConvolution(in_channels, hid_channels)
        self.gc_hids = nn.ModuleList([GraphConvolution(hid_channels, hid_channels) for _ in range(layer_num-2)])
        self.gc_out = GraphConvolution(hid_channels, out_channels)
        if activation == 'relu':
            self.activate = F.relu
        elif activation == 'sigmoid':
            self.activate = torch.sigmoid
        elif activation == 'tanh':
            self.activate =  torch.tanh

        
        self.pair_norm = pair_norm
        if pair_norm == True:
            self.norm = PairNorm()
        

        self.dropout = nn.Dropout(dropout)
        
        # for ppi dataset
        self.linear_out = nn.Linear(out_channels, 121)

    def forward(self, x, adj, task='nodecls', edges=None, ppi=False):
        x = self.gc_inp(x, adj)
        x = self.activate(x)

        for gc_layer in self.gc_hids:
            x = self.dropout(x)
            x = gc_layer(x, adj)
            
            if self.pair_norm:
                x = self.norm(x)
            
            x = self.activate(x)

        x = self.dropout(x)
        x = self.gc_out(x, adj)

        if task == 'nodecls':
            if ppi == False:
                # x.shape = [node_num, label_class_num]
                return F.log_softmax(x, dim=1)
            else:
                x = self.linear_out(x)
                # x.shape = [node_num, label_dim]
                return x
        elif task == 'linkpred':
            # x.shape = [node_num, hid_channels]
            assert edges != None
            src = x[edges[0]] # shape = [src_num, hid_channels]
            dst = x[edges[1]] # shape = [node_num, hid_channels]
            inner_prods = (src * dst).sum(dim=-1) # shape =[src_num]
            return inner_prods




