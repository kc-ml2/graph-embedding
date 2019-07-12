from models.gcn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv as GCN_lib
import pdb

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
        super().__init__()

        if use_package:
            self.conv1 = GCN_lib(in_features, dim_hidden, bias = bias)
            self.conv2 = GCN_lib(dim_hidden, out_class, bias = bias)
            self.inner_conv = GCN_lib(dim_hidden, dim_hidden, bias = bias)
        else:
            self.conv1 = GCNConv(in_features, dim_hidden, bias = bias)
            self.conv1 = GCNConv(dim_hidden, out_class, bias = bias)
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
        return F.log_softmax(x, dim = 1)

"""
    Implementation by Kipf
    dealing with large graphs by using sparse matrix in scipy
"""
class GCNModel(nn.Module):


    def __init__(self, in_features, dim_hidden, n_hidden_layer, out_class, bias):
        super(GCNModel, self).__init__()
        # Encoding input channels
        self.gcn1 = GCNConv(in_features, dim_hidden, bias = bias)

        # Decode to output channels
        self.gcn2 = GCNConv(dim_hidden, out_class, bias = bias)

        # hidden layers if more than 2 layers are needed
        self.inner_conv = GCNConv(dim_hidden, dim_hidden, bias = bias)
        self.n_hidden_layer = n_hidden_layer

    def forward(self, x, edge_index):

        # input to hidden
        x = self.gcn1(x, edge_index)
        pdb.set_trace()
        # activate
        x = F.relu(x)
        # dropout
        x = F.dropout(x, training = self.training)
        # if more than 2 layers, below layers would be passed.
        if self.n_hidden_layer > 2:
            for layer in range(self.n_hidden_layer - 2):
                x = self.inner_conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training = self.training)
        pdb.set_trace()
        # hidden to output
        x = self.gcn2(x, edge_index)
        pdb.set_trace()

        # return log_softmax
        return F.log_softmax(x, dim = 1)