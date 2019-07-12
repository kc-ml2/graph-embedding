from models.gcn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv as GCN_lib
import pdb
import traceback

"""
    Unweighted Graph GCN Implementation
"""

"""
    PyG implementation of GCN Code
    dealing with large graphs by using message passing idea, instead of using adjacency matrix
"""

# FIXME: Can implement 2 classes into 1 class by using conditional statement. Fix tomorrow.

class GCNNet(nn.Module):


    def __init__(self, in_features, dim_hidden, n_hidden_layer, out_class, bias, use_package):
        assert n_hidden_layer >= 2, "Number of Hidden Layer in GCN must be at least 2"
        super().__init__()

        if use_package:
            self.conv1 = GCN_lib(in_features, dim_hidden, bias = bias)
            self.conv2 = GCN_lib(dim_hidden, out_class, bias = bias)
            self.inner_conv = GCN_lib(dim_hidden, dim_hidden, bias = bias)
        else:
            self.conv1 = GCNConv(in_features, dim_hidden, bias = bias)
            self.conv2 = GCNConv(dim_hidden, out_class, bias = bias)
            self.inner_conv = GCNConv(dim_hidden, dim_hidden, bias = bias)
        self.n_hidden_layer = n_hidden_layer

    def forward(self, x, edge_index):
        # send input to hidden layer
        x = self.conv1(x, edge_index)
        # activate
        x = F.relu(x)
        #dropout
        x = F.dropout(x, training = self.training)
        # if more than 2 layers, below layers would be passed.
        if self.n_hidden_layer > 2:
            for layer in range(self.n_hidden_layer - 2):
                x = self.inner_conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training = self.training)
        
        # last layer
        x = self.conv2(x, edge_index)

        # log_softmax last layer
        # FIXME: Only work for exclusive label like Cora; Not working for PPI(Multiple label for one data)
        return F.log_softmax(x, dim = 1)