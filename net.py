# for overall network
import argparse
import setting
import torch
import torch.optim as optim
from models.GCNmodel import GCNNet
import numpy as np
from torch_geometric.data import DataLoader
from utils.datagenerator import datagenerator

#processing the network 
from processing.train import train
from processing.val import val
from processing.test import test

# for logging
import time
import traceback

# for saving
import os.path
from pathlib import Path
import io

# for debug
import pdb

"""
    function that run network and learn
    Composed by reading data / setting model / train / val / test
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
    process = args.process
    use_package = args.use_package_implementation
    del args

    # Read the Data
    # Only internal datasets of pyG are available currently
    dataset = datagenerator(data_class, dataname = data_name)
    train_dataset = dataset.train_dataset
    val_dataset = dataset.val_dataset
    test_dataset = dataset.test_dataset
    data_name = dataset.data_name
    if data_name == "PPI":
        assert loss_ft == "bce" or  loss_ft == "mse", "PPI cannot work with other losses"
    elif data_name == "Cora":
        assert loss_ft == "nll" or  loss_ft == "cross_entropy", "Cora cannot work with other losses"
    del dataset
    try:
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

        if process != 'test':
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

            if process == 'whole':
            # Test
                test(model = model_running, test_data = test_dataset, device = device, logger = logger)
            
            # Save model
            elif process == 'train':
                Path(os.path.join(setting.MODEL_SAVED_PATH, data_name)).mkdir(parents = True, exist_ok = True)
                filename = os.path.join(setting.MODEL_SAVED_PATH, data_name, "trained.pt")
                torch.save(model_running.state_dict(), filename)

            # Error
            else:
                logger.log("invalid input for args.process", "ERROR")
        
        elif process == "test":
            try:
                filename = os.path.join(setting.MODEL_SAVED_PATH, data_name ,"trained.pt")
                if os.path.isfile(filename):
                    model_running.load_state_dict(torch.load(filename))
                    test(model = model_running, test_data = test_dataset, device = device, logger = logger)
                else:
                    raise IOError('No Trained Model')
                
            except Exception:
                logger.log(traceback.format_exc(), "ERROR")
        
        logger.log("Done!", 'INFO')
                

    except Exception:
        logger.log(traceback.format_exc(), "ERROR")

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