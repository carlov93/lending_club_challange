import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch


class DataSetLoans(Dataset):
    """
    Dataset an abstract class representing a dataset. The sensor data inherit Dataset and
    override the following methods.
    """
    def __init__(self, X, y):
        self.y = [np.array(i) for i in y]
        self.X = X

    def __getitem__(self, index):
        """
        This method reads the data from the csv file. It is memory efficient because all rows
        are not stored in the memory at once but read as required.
        """
        x = self.X[index]
        y = self.y[index]
        x_torch = torch.from_numpy(x)
        y_torch = torch.from_numpy(y)
        return (x_torch.float(), y_torch.float()) 

    def __len__(self):
        return self.X.shape[0]