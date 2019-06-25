import os.path as osp

import torch
from torch_geometric.data import Dataset
import setting
import collections
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import torch_geometric.utils as utils
import torch_geometric

def to_list(x):
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
        x = [x]
    return x

def files_exist(files):
    return all([osp.exists(f) for f in files])


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return ['youtube.txt']

    @property
    def processed_file_names(self):
        return ['data_0.pt', 'data_1.pt', ...]
    
    def __len__(self):
        return len(self.processed_file_names)

    @property
    def processed_paths(self):
        print(self.root)
        return [osp.join(self.root, 'youtube.txt')]

    @property
    def raw_paths(self):
        return [osp.join(self.root, 'youtube.txt')]
    
    def download(self):
        pass

    def process(self):
        #pass
        #print(self.root)
        #print(self.processed_paths)
        #print(files_exist(self.processed_paths))
        
        i = 0
        for raw_path in self.raw_paths:
            print('raw_path = {}'.format(raw_path))
            # Read data from `raw_path`.
            G = nx.read_edgelist(raw_path, delimiter = '\t', nodetype=int, edgetype = int, data = False)
            print(nx.info(G))
            edge_index = torch.tensor(list(G.edges)).t().contiguous() 
            # print(G.nodes(data = True))

            # print(type(list(G.nodes(data=False))))
            # print(type(list(G.nodes(data=False))[0]))
            keys = []
            keys += list(G.nodes(data=False))
            print('node sucess!')
            keys += list(list(G.edges(data=True))[0][2].keys())
            data = {key: [] for key in keys}

            for _, feat_dict in G.nodes(data=True):
                for key, value in feat_dict.items():
                    data[key].append(value)

            for _, _, feat_dict in G.edges(data=True):
                for key, value in feat_dict.items():
                    data[key].append(value)

            for key, item in data.items():
                data[key] = torch.tensor(item)
            data['edge_index'] = edge_index

            data = torch_geometric.data.Data.from_dict(data)
            
            print(0)
            # data = utils.from_networkx(G)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, ops.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1
        
        # pass
    
                

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

youtube = MyOwnDataset(setting.YOUTUBE_DATASET)
youtube.process()