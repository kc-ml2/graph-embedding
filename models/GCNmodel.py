from models.gcn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv as GCN_lib

"""
    Unweighted Graph GCN Implementation
"""

class GCNNet(nn.Module):
    def __init__(self, in_features: int, num_hidden: int, out_class: int):
        super(GCNNet, self).__init__()
        self.conv1 = GCN_lib(in_features, num_hidden)
        self.conv2 = GCN_lib(num_hidden, out_class)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GCNModel(nn.Module):


    def __init__(self, in_features, num_hidden, out_class, dropout):
        super(GCNModel, self).__init__()

        self.gcn1 = GCNConv(in_features, num_hidden)
        self.gcn2 = GCNConv(num_hidden, out_class)
        self.dropout = dropout

    def forward(self, X, edge_index):
        X = self.gcn1(X, edge_index)
        X = F.relu(self.gcn1(X,edge_index))
        X = F.dropout(X, self.dropout, training = self.training)
        X = self.gcn2(X, edge_index)

        return F.log_softmax(X, dim = 1)