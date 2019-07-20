# for path
import sys
from pathlib import Path
sys.path.append((Path(__file__).parent.resolve().parent.as_posix()))

# for inpackage data
import importlib
from setting import *
import inspect

#for DB

#for print
import traceback

# for debug
import pdb

"""
    Class that read data
    InPackage / DB / External 
"""

class datagenerator:


    def __init__(self, dataclass: str, **kwargs):
        m = importlib.import_module("torch_geometric.datasets")
        datasets = inspect.getmembers(m, inspect.isclass)
        datasets_names  = list(list(zip(*datasets))[0])
        # Data is in Package
        if dataclass in datasets_names:
            dataset_NameClassTuple = datasets[datasets_names.index(dataclass)]
            try:
                if 'dataname' in kwargs:
                    self.get_data(dataset_NameClassTuple, type = "train", dataname = kwargs['dataname'])
            except Exception:
                print(traceback.format_exc)
        else:
            """
                Search whether the data is in DB or External
            """
            pass
            
    def get_data(self, dataset_NameClassTuple, **kwargs):
        try:
            dataset_name = dataset_NameClassTuple[0]
            dataset_class = dataset_NameClassTuple[1]
            if dataset_name is "PPI":
                self.train_dataset = dataset_class(DATA_PATH, split = 'train')
                self.val_dataset = dataset_class(DATA_PATH, split = 'val')
                self.test_dataset = dataset_class(DATA_PATH, split = 'test')
                self.data_name = "PPI"
            elif dataset_name is "Planetoid":
                try:
                    self.train_dataset = dataset_class(DATA_PATH, name = kwargs['dataname'])
                    self.val_dataset = dataset_class(DATA_PATH, name = kwargs['dataname'])
                    self.test_dataset = dataset_class(DATA_PATH, name = kwargs['dataname'])
                    self.data_name = kwargs['dataname']
                except Exception:
                    print(traceback.format_exc())
            else:
                raise NotImplementedError('Not Implemented!')
        except Exception:
            print(traceback.format_exc())
        
            
    