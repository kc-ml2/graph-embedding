import argparse
import setting
import json
from net import run_network

def main():
    # Read argument
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cuda', action = 'store_true', help = 'using GPU?')
    parser.add_argument('--fastmode', action = 'store_true', help = 'validate during training pass')
    parser.add_argument('--seed', type = int, default = 42, help = 'random seed')
    parser.add_argument('--epochs', type = int, default = 200, help = 'num of train epochs')
    parser.add_argument('--lr', type = float, default = 0.01, help = 'initial learning rate')
    parser.add_argument('--loss_ft', type = str, default = "nll", help = 'loss function. nll / cross-entropy')
    parser.add_argument('--L1_regular', action = 'store_true', help = 'use L1 regularization?\
                        if not, L2')
    parser.add_argument('--weight_decay',type = float, default = 5e-4, help = 'weight decay')
    parser.add_argument('--hidden', type = int, default = 16, help = 'number of hidden layers')
    parser.add_argument('--dropout', type = float, default = 0.5, help = 'dropout rate(1 - prob)')
    parser.add_argument('--data', type = str, default = 'default', help = 'dataset to use')
    parser.add_argument('--model', type = str, default = 'GCN', help = 'GraphML model to use')

    args = parser.parse_args([])

    # Read config written by json file
    with open('config.json', 'r') as f:
        config = json.load(f)
    for key, item in config['DEFAULT'].items():
        setattr(args, key, item)

    # part where run the network
    run_network(args)

if __name__ == '__main__':
    main()