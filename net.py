import argparse
import setting
import torch
import torch.optim as optim
from models.GCNmodel import GCNModel
from models.GCNmodel import GCNNet
from torch_geometric.datasets import PPI
from torch_geometric.datasets import CoMA
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
import time
import traceback
import utils

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

    # settings from args
    np.random.seed(args.seed); torch.manual_seed(args.seed);
    device = torch.device('cuda' if cuda else 'cpu');
    model = args.model; hidden = args.hidden; dropout = args.dropout;
    lr = args.lr; weight_decay = args.weight_decay; loss_ft = args.loss_ft
    epochs = args.epochs; data_name = args.data

    try:
        # No data Implemented yet
        if data_name == 'PPI':
            train_dataset = PPI(root = setting.DATA_PATH, split = "train")
            val_dataset = PPI(root = setting.DATA_PATH, split = "val")
            test_dataset = PPI(root = setting.DATA_PATH, split = "test")

        elif data_name == 'Cora':
            train_dataset = Planetoid(setting.DATA_PATH, name = 'Cora')
            val_dataset = train_dataset
            test_dataset = train_dataset

        else:
            # TODO: Implement the GraphDB Pt
            raise IOError('No Data')

        if model == "GCN":
            model_running = GCNNet(in_features = train_dataset.num_features,\
                num_hidden = hidden, out_class = train_dataset.num_classes).to(device)
        else:
            raise NotImplementedError('Not Implemented!')

        # load data
        train_loader = DataLoader(train_dataset, batch_size = 1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = True)

        #determine whether data is all full or splitted
        isfull = (train_dataset == val_dataset)

        print("read Success")

        # train model
        start_time = time.time()
        """
        if model == "GCN":
            model_running = GCNNet(in_features = train_dataset.num_features,\
                 num_hidden = hidden, out_class = train_dataset.num_classes).to(device)
        else:
            raise NotImplementedError('Not Implemented!')
        """

        # set optimizer
        optimizer = optim.Adam(params = model_running.parameters(), lr = lr, weight_decay = weight_decay)

        for epoch in range(epochs):

            # train
            for train_batch in train_loader:
                train_batch.to(device)
                train(epoch = epoch, batch = train_batch, model = model_running,\
                     loss_ft = loss_ft, optimizer = optimizer, is_full_data = isfull)

            # val
            for val_batch in val_loader:
                val_batch.to(device)
                val(epoch = epoch, batch = val_batch, model = model_running, loss_ft = loss_ft, is_full_data = isfull)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - start_time))

        # test
        
        test(model = model_running, test_data = test_dataset, device = device)
        print("done!")

    except Exception:
        print(traceback.format_exc())

"""
    Train the model
"""

def train(epoch, batch, model, loss_ft, optimizer, is_full_data = False):
    model.train()
    optimizer.zero_grad()
    # run model
    log_prob = model(batch.x, batch.edge_index)
    
    if is_full_data:
        log_prob = log_prob[batch.train_mask]
        label = batch.y[batch.train_mask]
    else:
        label = batch.y
    try:
        if loss_ft == 'nll':
            loss = F.nll_loss(log_prob, label.long())
        elif loss_ft == 'cross_entropy':
            loss = F.cross_entropy(log_prob, label.long())
        elif loss_ft == 'mse':
            loss = F.mse_loss(log_prob, label)
        elif loss_ft == 'bce':
            log_prob = torch.exp(log_prob)
            loss = F.binary_cross_entropy(log_prob, label)
        else:
            raise NotImplementedError('Not implemented')
        loss.backward()
        optimizer.step()
        print("Epoch {} | Loss {:.4f}".format(\
                epoch, loss.item()))
    
    except Exception:
        print(traceback.format_exc())

def val(epoch, batch, model, loss_ft, is_full_data = False):
    model.eval()
    # put into model
    log_prob = model(batch.x, batch.edge_index)
    if is_full_data:
        log_prob = log_prob[batch.train_mask]
        label = batch.y[batch.train_mask]
    else:
        label = batch.y
    try:
        if loss_ft == "nll":
            loss_val = F.nll_loss(log_prob, label.long())
        elif loss_ft == "cross_entropy":
            loss_val = F.cross_entropy(log_prob, label.long())
        elif loss_ft == "mse":
            loss_val = F.mse_loss(log_prob, label)
        elif loss_ft == 'bce':
            log_prob = torch.exp(log_prob)
            loss_val = F.binary_cross_entropy(log_prob, label)
        else:
            raise NotImplementedError('Not Implemeted')
        print(log_prob.shape)
        print(label.shape)
        acc = utils.accuracy(log_prob, label)
        #acc = 0
        print('epoch : {} || loss_val : {:.4f} || Accuracy: {:.4f}'.format(epoch, loss_val, acc))
    except Exception:
        print(traceback.format_exc())

def test(model, test_data, device):

    # merge test data / model data
    output = []
    label = []
    for data in test_data:
        data.to(device)
        log_p = model(data.x, data.edge_index)
        output.append(log_p)
        label.append(data.y)
    result = torch.cat(output, dim = 0)
    answer = torch.cat(label, dim = 0)
    result = torch.sigmoid(result)
    
    # calculate accuracy
    acc = utils.accuracy(result, answer)
    print("Final Accuracy is = {}".format(acc))

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