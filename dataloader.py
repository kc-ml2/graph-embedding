import torch
from torch_geometric.data import Dataset
import setting
import os.path as osp
from torch_geometric.data import Data
"""
    Not Implemented Yet!
"""
class YoutubeDataset(Dataset):


    def __init__(self, root = setting.YOUTUBE_DATASET, transform = None, pre_transform = None):
        super(YoutubeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['youtube.txt']

    @property
    def processed_file_names(self):
        return ['data_0.pt']

    def __len__(self):
        return len(self.processed_file_names)

    @property
    def processed_paths(self):
        return  [osp.join(self.root, 'youtube.txt')]


    def download(self):
        
    def process(self):
