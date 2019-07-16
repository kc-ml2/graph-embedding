# for overall network
import argparse
import setting
import torch
import torch.optim as optim
from models.GCNmodel import GCNNet
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
from utils import utils
from utils.datagenerator import datagenerator

# for print
import time
import traceback

# for debug
import pdb

"""
    function that run network and learn
    Composed by reading data / setting model / train / val
"""

def run_network(args, logger):
    
    # set CPU or GPU
    if not torch.cuda.is_available():
        args.cuda = False
        logger.INFO("Cuda is not available. Automatically set Cuda False")
    cuda = args.cuda
    device = torch.device('cuda' if cuda else 'cpu')

    # bring settings from args
    np.random.seed(args.seed); torch.manual_seed(args.seed);
    device = torch.device('cuda' if cuda else 'cpu');
    model = args.model; dim_hidden = args.dim_hidden; 
    n_hidden_layer = args.n_hidden_layer; lr = args.lr; 
    weight_decay = args.weight_decay; loss_ft = args.loss_ft
    epochs = args.epochs; batch_size = args.batch_size; bias = args.bias
    data_class = args.data; data_name = args.data_name
    use_package = args.use_package_implementation
    del args

    # Read the Data
    dataset = datagenerator(data_class, dataname = data_name)
    train_dataset = dataset.train_dataset
    val_dataset = dataset.val_dataset
    test_dataset = dataset.test_dataset
    data_name = dataset.data_name
    if data_name == "PPI":
        assert loss_ft == "bce" or  loss_ft == "mse", "PPI cannot work with other losses"
    elif data_name == "Cora":
        assert loss_ft == "nll" or  loss_ft == "cross_entropy", "PPI cannot work with other losses"
    del dataset
    try:
        # Only internal datasets of pyG are available currently
        """
        if data_name == 'PPI':
            assert loss_ft == 'bce', "PPI only works with bce loss"
            train_dataset = PPI(root = setting.DATA_PATH, split = "train")
            val_dataset = PPI(root = setting.DATA_PATH, split = "val")
            test_dataset = PPI(root = setting.DATA_PATH, split = "test")
            logger.log("The PPI Data may not have been fully optimized yet", "WARNING")
        # Default
        elif data_name == 'Cora':
            train_dataset = Planetoid(setting.DATA_PATH, name = 'Cora')
            val_dataset = train_dataset
            test_dataset = train_dataset

        else:
            # TODO: Implement the Outer Source of data
            raise IOError('No Data')
        """
        # Input of Model
        if model == "GCN":
            # Run GCN
            model_running = GCNNet(in_features = train_dataset.num_features, \
                dim_hidden = dim_hidden, n_hidden_layer = n_hidden_layer, \
                    out_class = train_dataset.num_classes, bias = bias, \
                        use_package = use_package).to(device)
        else:
            #Would implement other models later
            raise NotImplementedError('Not Implemented!')

        # load data
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

        # determine whether train and val splitted.
        isfull = (train_dataset == val_dataset)

        # Print the process
        logger.log("Read Success!", "INFO")

        # Training and Validation Start
        start_time = time.time()
        # set optimizer
        optimizer = optim.Adam(params = model_running.parameters(), lr = lr, weight_decay = weight_decay)

        # Loop the epoch
        for epoch in range(epochs):

            # train
            for train_batch in train_loader:
                train_batch.to(device)
                train(epoch = epoch, batch = train_batch, model = model_running,\
                     loss_ft = loss_ft, optimizer = optimizer, logger = logger, is_full_data = isfull)

            # val
            for val_batch in val_loader:
                val_batch.to(device)
                val(epoch = epoch, batch = val_batch, model = model_running, \
                    loss_ft = loss_ft, logger = logger, is_full_data = isfull)

        # Finish the Model Training
        logger.log("Optimization Finished!", "INFO")
        logger.log("Total time elapsed: {:.4f}s".format(time.time() - start_time), "INFO")

        # Test
        test(model = model_running, test_data = test_dataset, device = device, logger = logger)
        logger.log("Done!", 'INFO')

    except Exception:
        logger.log(traceback.format_exc(), "ERROR")

"""
    Train the model
"""

def train(epoch, batch, model, loss_ft, optimizer, logger, is_full_data = False) -> None:
    model.train()
    optimizer.zero_grad()

    # run model
    log_prob = model(batch.x, batch.edge_index)
    
    # Split the data if data is not splitted into train and val
    if is_full_data:
        log_prob = log_prob[batch.train_mask]
        label = batch.y[batch.train_mask]
    else:
        label = batch.y
    try:
        # Select Loss ft
        if loss_ft == 'nll':
            loss = F.nll_loss(log_prob, label.long())
        elif loss_ft == 'cross_entropy':
            loss = F.cross_entropy(log_prob, label.long())
        elif loss_ft == 'mse':
            loss = F.mse_loss(log_prob, label)
        elif loss_ft == 'bce':
            # bce loss have to be parsed to 0~1
            log_prob = torch.exp(log_prob)
            loss = F.binary_cross_entropy(log_prob, label)
        else:
            raise NotImplementedError('Not implemented')

        # Back propagate and learn
        loss.backward()
        optimizer.step()

        #print losses
        logger.log("Epoch {} | Loss {:.4f}".format(\
                epoch, loss.item()), "TRAIN")
    
    except Exception:
        logger.log(traceback.format_exc(), "ERROR")

"""
    Validate the Model
"""

def val(epoch, batch, model, loss_ft, logger, is_full_data = False) -> None:
    model.eval()
    # put into model
    log_prob = model(batch.x, batch.edge_index)

    # split data if data is not splitted into train and val
    if is_full_data:
        log_prob = log_prob[batch.train_mask]
        label = batch.y[batch.train_mask]
    else:
        label = batch.y
    try:
        # Select loss ft
        if loss_ft == "nll":
            loss_val = F.nll_loss(log_prob, label.long())
        elif loss_ft == "cross_entropy":
            loss_val = F.cross_entropy(log_prob, label.long())
        elif loss_ft == "mse":
            loss_val = F.mse_loss(log_prob, label)
        elif loss_ft == 'bce':
            # bce loss requires additional parsing to 0~1
            log_prob = torch.exp(log_prob)
            loss_val = F.binary_cross_entropy(log_prob, label)
        else:
            raise NotImplementedError('Not Implemeted')
        # Calculate accuracy
        acc = utils.accuracy(log_prob, label)
        # pdb.set_trace()
        logger.log('epoch : {} || loss_val : {:.4f} || Accuracy: {:.4f}'.format(epoch, loss_val, acc), "VAL")
    except Exception:
        logger.log(traceback.format_exc(), "ERROR")

"""
    Test the Model
"""
def test(model, test_data, device, logger) -> None:

    # merge test data & model data
    output = []
    label = []
    for data in test_data:
        data.to(device)
        log_p = model(data.x, data.edge_index)
        output.append(log_p)
        label.append(data.y)

    # prediction
    result = torch.cat(output, dim = 0)
    result = torch.sigmoid(result)
    # answer
    answer = torch.cat(label, dim = 0)
    
    # calculate accuracy
    acc = utils.accuracy(result, answer)
    logger.log("Final Accuracy is = {}".format(acc), "TEST")
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