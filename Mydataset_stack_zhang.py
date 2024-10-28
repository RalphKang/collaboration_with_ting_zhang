import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from PIL import Image
from torch.utils.data.dataset import Dataset
import os
import random
import scipy.io


"""NOTE:
Process Qingjie's dataset KRY 24th,10, 2024
@author KRY"""

# filenameToPILImage = lambda x: Image.open(x)

def fix_random_seeds(seed=42):
    """
    Fix random seeds for reproducibility in Python, NumPy, and PyTorch.
    
    Args:
    seed (int): The seed to use for random number generators.
    """
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed for CPU
    torch.manual_seed(seed)
    
    # If using CUDA, set seeds for CUDA as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Ensure deterministic behavior for certain PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class dataset_zhang_bio(Dataset):
    def __init__(self, args,data_ratio,train_mode):
        """
        this module is used to load data for training SUPERVISED MODEL
        where, input and labels are normalized

        The data contains all input related to the loss function: including x, z, m, m0, u0

        """
        # input related data---------------------------------------------------------------------
        super(dataset_zhang_bio, self).__init__()
        whole_data=np.load(args.dataset_dir)
        data=whole_data[:,:3]
        data=torch.from_numpy(data)
        output=whole_data[:,3]
        output=torch.from_numpy(output).unsqueeze(1)
        sam_num=len(data)
        sam_pos,sam_neg=np.meshgrid(np.arange(sam_num),np.arange(sam_num))
        sam_pos=sam_pos.reshape(-1)
        sam_neg=sam_neg.reshape(-1)

        input=torch.cat((data[sam_pos],data[sam_neg]),dim=1)
        label=output[sam_pos]-output[sam_neg]
        # negative label is 0, positive label is 1
        label[label<0]=0
        label[label>0]=1

        self.input_max = torch.max(input,dim=0)[0]+0.001
        self.input_min = torch.min(input,dim=0)[0]-0.001
        self.input_max=self.input_max.float()
        self.input_min=self.input_min.float()
        input = (input - self.input_min) / (self.input_max - self.input_min) # data normalization
        self.orig_input = input

        self.label_max = torch.max(label,dim=0)[0]
        self.label_min = torch.min(label,dim=0)[0]
        label = (label - self.label_min) / (self.label_max - self.label_min)
        self.orig_label = label
        input_length=input.shape[0]
        self.shuffle_index=np.arange(len(input))
        self.shuffle_index=np.random.permutation(self.shuffle_index) # change the order of the data

        data_amount=int(len(input)*data_ratio)

        if train_mode: # if train mode is True, the data is used for training
            data_index=self.shuffle_index[:data_amount]
            self.input=input[data_index]
            self.label=label[data_index]
            self.length=len(data_index)

        else:
            data_index=self.shuffle_index[-data_amount:]
            self.input=input[data_index]
            self.label=label[data_index]
            self.length=len(data_index)
    def __len__(self):
        """
        :return: the length of the dataset
        """
        return self.length


    def __getitem__(self, index):
        input=self.input[index]
        label=self.label[index]
        
        return input.float(), label.float()
    