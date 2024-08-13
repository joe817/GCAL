import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, save_mem=True, use_bn=True):
        super(GCN, self).__init__() 

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]): 
            x = conv(x, edge_index, edge_weight)
            if self.use_bn:
                x = self.bns[i](x) 
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logit = self.convs[-1](x, edge_index, edge_weight)
        return logit, x

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, use_bn=True):
        super(SAGE, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logit = self.convs[-1](x, edge_index)
        return logit, x
    

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2):
        super(GAT, self).__init__()

        self.heads = heads
        self.hidden_channels = hidden_channels

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True) ) 
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False))

        self.dropout = dropout
        self.activation = F.elu 

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logit = self.convs[-1](x, edge_index, edge_weight)
        x = x.reshape(x.shape[0], self.heads, self.hidden_channels)
        x = x.mean(dim=1)
        return logit, x


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5):
        super(GIN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GINConv(nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
            ), train_eps=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GINConv(nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                ), train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GINConv(nn.Sequential(
                nn.Linear(hidden_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ), train_eps=True))

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
