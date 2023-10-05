import nibabel as nib
import torch
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset
from torchvision import transforms 

# load the slices
class ExampleDataset(Dataset):
    """ SliceLoader 
    
    A class which is used to allow for efficient data loading of the training data. 
    Args:
        - torch.utils.data.Dataset: A PyTorch module from which this class inherits which allows it 
        to make use of the PyTorch dataloader functionalities. 
    
    """
    def __init__(self, train=False, val=False, test=False, N=16):
        #4632
        """ Class constructor
        
        Args:
            - downsampling_factor: The factor by which the loaded data has been downsampled. 
            - N: The length of the dataset. 
            - folder_name: The folder from which the data comes from 
            - is_train: Whether or not the dataloader is loading training data (and therefore randomised data).   
        """ 
        self.val = val 
        self.test = test 
        self.train = train
        self.test = test 
        self.N = N - 1

        
    def __len__(self):
        """ __len__
        
        A function which configures and returns the size of the datset. 
        
        Output: 
            - N: The size of the dataset. 
        """
        return (self.N)    
    
    def __getitem__(self, idx):
        if self.train: 
            image = torch.tensor(np.random.rand(1, 128, 128).astype(np.float32))
            label = np.random.randint(0, 2) 
            
        elif self.val: 
            image = np.random.rand(1, 128, 128).astype(np.float32)
            label = np.random.randint(0, 2)
        
        elif self.test:
            image = np.random.rand(1, 128, 128).astype(np.float32)
            label = np.random.randint(0, 2) 
            
        return {'img': image, 'index': idx, 'label': label}    
    

