import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj, dense_to_sparse

from nets import *

class Model(nn.Module):
    def __init__(self, args, feature_dim, num_labels):
        super(Model, self).__init__()
        gnn = args.model
        if gnn == 'gcn':
            self.gnn = GCN(in_channels=feature_dim,
                        hidden_channels=args.hidden_channels,
                        out_channels=num_labels,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=not args.no_bn)
        elif gnn == 'sage':
            self.gnn = SAGE(in_channels=feature_dim,
                        hidden_channels=args.hidden_channels,
                        out_channels=num_labels,
                        num_layers=args.num_layers,
                        dropout=args.dropout)
        elif gnn == 'gat':
            self.gnn = GAT(in_channels=feature_dim,
                        hidden_channels=args.hidden_channels,
                        out_channels=num_labels,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        heads=args.gat_heads)
        elif gnn == 'gin':
            self.gnn = GIN(in_channels=feature_dim,
                        hidden_channels=args.hidden_channels,
                        out_channels=num_labels,
                        num_layers=args.num_layers,
                        dropout=args.dropout)

        self.args = args

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index, y, train_mask, edge_weight=None):
        out, _ = self.gnn(x, edge_index, edge_weight)
        loss = self.sup_loss(y[train_mask], out[train_mask])
        return loss
    
    def sup_loss(self, y, out):
        if self.args.dataset in ('twitch', 'fb100', 'elliptic'):
            y  = F.one_hot(y, y.max()+1).float()
            loss = F.binary_cross_entropy_with_logits(out, y)
        else:
            out = F.log_softmax(out, dim=1)
            loss = F.nll_loss(out, y, ignore_index=-1)

        return loss



    def inference(self, x, edge_index, edge_weight=None):
        out, _ = self.gnn(x, edge_index, edge_weight)
        out = F.softmax(out, dim=1)
        return out

    def embed(self, x, edge_index, edge_weight=None):
        _, x = self.gnn(x, edge_index, edge_weight)
        return x







