import argparse
import setting
import torch
import torch.optim as optim
from models.GCNmodel import GCNModel
from models.GCNmodel import GCNNet
from torch_geometric.datasets import PPI
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
import time
import traceback
import utils

"""
    Train the model
"""

def train(epoch, batch, model, loss_ft, optimizer):
    model.train()
    optimizer.zero_grad()
    log_prob = model(batch.x, batch.edge_index)
    try:
        if loss_ft == 'nll':
            loss = F.nll_loss(log_prob, batch.y.long())
        elif loss_ft == 'cross_entropy':
            loss = F.cross_entropy(log_prob, batch.y.long())
        elif loss_ft == 'mse':
            loss = F.mse_loss(log_prob, batch.y)
        elif loss_ft == 'bce':
            log_prob = torch.sigmoid(log_prob)
            loss = F.binary_cross_entropy(log_prob, batch.y)
        else:
            raise NotImplementedError('Not implemented')
        loss.backward()
        optimizer.step()
        print("Epoch {} | Loss {:.4f}".format(\
                epoch, loss.item()))
    
    except Exception:
        print(traceback.format_exc())

def val(epoch, batch, model, loss_ft):
    model.eval()
    log_prob = model(batch.x, batch.edge_index)
    try:
        if loss_ft == "nll":
            loss_val = F.nll_loss(log_prob, batch.y.long())
        elif loss_ft == "cross_entropy":
            loss_val = F.cross_entropy(log_prob, batch.y.long())
        elif loss_ft == "mse":
            loss_val = F.mse_loss(log_prob, batch.y)
        elif loss_ft == 'bce':
            log_prob = torch.sigmoid(log_prob)
            loss_val = F.binary_cross_entropy(log_prob, batch.y)
        else:
            raise NotImplementedError('Not Implemeted')
        acc = utils.accuracy(log_prob, batch.y)
        """
        log_prob = (log_prob>0.5).float()
        # print(log_prob)
        correct = (log_prob == batch.y).float().sum()
        # print(correct)
        acc = correct / log_prob.shape[0]
        """
        # pred = log_prob.max(dim = 1)[0]
        # print(pred)
        #correct = float (pred.eq(batch.y).sum().item())
        #acc = correct / batch.num_nodes
        # acc = 0
        print('epoch : {} || loss_val : {:.4f} || Accuracy: {:.4f}'.format(epoch, loss_val, acc))
    except Exception:
        print(traceback.format_exc())

"""
    function that run network and learn
    Composed by reading data / setting model / train / val
"""

def run_network(args):
    
    # set CPU or GPU
    if not torch.cuda.is_available():
        args.cuda = False
        print("Cuda is not available. Automatically set Cuda False")
    cuda = args.cuda    
    device = torch.device('cuda' if cuda else 'cpu')

    try:
        # No data Implemented yet
        if args.data == 'default':
            train_dataset = PPI(root = setting.DATA_PATH, split = "train")
            val_dataset = PPI(root = setting.DATA_PATH, split = "val")
        else:
            # TODO: Implement the GraphDB Pt
            raise IOError('No Data')
        # load data
        train_loader = DataLoader(train_dataset, batch_size = 2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size = 2, shuffle = True)
        print("read Success")

        # settings from args
        np.random.seed(args.seed); torch.manual_seed(args.seed);
        device = torch.device('cuda' if cuda else 'cpu');
        model = args.model; hidden = args.hidden; dropout = args.dropout;
        lr = args.lr; weight_decay = args.weight_decay; loss_ft = args.loss_ft

        # train model
        start_time = time.time()
        if model == "GCN":
            model_running = GCNNet(in_features = train_dataset.num_features, num_hidden = hidden, out_class = train_dataset.num_classes).to(device)
            #model_running = GCNModel(in_features = train_dataset.num_features, num_hidden = hidden, out_class = train_dataset.num_classes\
            #    , dropout = dropout).to(device)
        else:
            raise NotImplementedError('Not Implemented!')

        # set optimizer
        optimizer = optim.Adam(params = model_running.parameters(), lr = lr, weight_decay = weight_decay)

        for epoch in range(args.epochs):
            for train_batch in train_loader:
                train(epoch = epoch, batch = train_batch, model = model_running,\
                     loss_ft = loss_ft, optimizer = optimizer)
            for val_batch in val_loader:
                val(epoch = epoch, batch = val_batch, model = model_running, loss_ft = loss_ft)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - start_time))
        print("done!")

    except Exception:
        print(traceback.format_exc())

"""

    for epoch:
        for train_batch in loader1:
            train(train_batch):
                ...
        for valid_batch in loader2:
            eval(valid_batch):
                ...

    for epoch:
        train(loader1):
            for train_batch in loader1:
                ...
        eval(loader2):
            for valid_batch in loader2:
                ...
"""

