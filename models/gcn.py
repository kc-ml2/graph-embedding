# for model
import torch
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# for matrix parsing
from utils.utils import edge_to_adj
import math

# for debug
import pdb

"""
    Unweighted Matrix Only!!
    In order to weighted martrix, fix utils.utils.edge_to_adj
"""

"""
    Implementation based on the code of Kipf
    "https://github.com/tkipf/pygcn"
    changed the part which receive the edge_index and convert to the adj matrix
    Deal with large graphs by using Sparse matrix in Scipy
"""

class GCNConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bias = True):
        super().__init__()

        # in, out, bias, weight dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    # Normalize
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # X * W
        support = torch.mm(input, self.weight)
        # Get adjacency matrix
        # Parsing A to hat{A} is conducted inside the edge_to_adj
        adj = edge_to_adj(input, edge_index)
        # FIXME: to get the config from args
        if torch.cuda.is_available:
            adj = adj.cuda()
        # hat{A} * (X * W)
        output = torch.spmm(adj, support)

        # hat{A} * (X * W) + B
        if self.bias is not None:
            return (output + self.bias)
        else:
            return output
