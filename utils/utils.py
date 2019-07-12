import pdb
import scipy.sparse as sp
import numpy as np
import torch

"""
    Calculating Accuracy
    p > 0.5 -> predict 1
    p <= 0.5 -> predict 0
"""

def accuracy(log_prob, labels):
    if log_prob.shape == labels.shape: 
        pred = (log_prob>0.5).float()
        print(pred[pred > 0])
        #print(pred)
        correct = (pred == labels).float().sum()
    else:
        _, pred = log_prob.max(dim=1)
        correct = float (pred.eq(labels).sum().item())
    return correct / len(labels)


"""
    edge list -> hat{A}
"""

def edge_to_adj(x, edges):
    edges = edges.cpu()
    x = x.cpu()
    #pdb.set_trace()
    # make sparse matrix
    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0, :], edges[1, :])),\
                        shape=(x.shape[0], x.shape[0]),\
                        dtype=np.float32)
    # Not fully understood....
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # hat{A} = A + I
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def normalize(mx):
    """Row-normalize sparse matrix"""
    # D^-1/2 * hat{A} * D^-1/2
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    # Exception handling
    r_inv[np.isinf(r_inv)] = 0.
    # making D
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    