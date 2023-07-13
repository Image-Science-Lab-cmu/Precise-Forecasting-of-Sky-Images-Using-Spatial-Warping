import h5py
import torch.utils.data as data
import numpy as np
import torch
from warp import *


## Hyperparameter
WarpUpSampFactor = 0.707
#############

class DatasetFromFolder(data.Dataset):
    def __init__(self, input_files, target_files):
        super(DatasetFromFolder, self).__init__()

        self.inputFile = h5py.File(input_files, 'r')
        self.targetFile = h5py.File(target_files, 'r')
        
        self.n_images = len(self.inputFile)

    def __getitem__(self, index):
        XfileName = 'X' + str(index)
        YfileName = 'y' + str(index)
        
        inputs = self.inputFile[XfileName]
        inputs = inputs.value
        inputs = np.float32(inputs)
        inputs = inputs/255
        #Warp Here
        inputs = warp(inputs, WarpUpSampFactor)
        inputs = np.moveaxis(inputs, 2, 0)
        inputs = torch.from_numpy(inputs)
        
        
        target = self.targetFile[YfileName]
        target = target.value
        target = np.float32(target)
        target = target/255
        #Warp Here
        target = warp(target, WarpUpSampFactor)
        
        target = np.moveaxis(target, 2, 0)
        target = torch.from_numpy(target)
        
        
        inputs = np.float32(inputs)
        target = np.float32(target)
        return inputs, target

    def __len__(self):
        return self.n_images