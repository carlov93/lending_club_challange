import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch


class DataPreperator:
    """
    This class provides methods for pre-processing the data
    """
    def __init__(self, pandas_file, ignored_features, stake_training_data,
                 features_not_to_scale):
        self.dataset = pandas_file
        self.scaler = StandardScaler()
        self.ignored_features = ignored_features
        self.features_not_to_scale = features_not_to_scale
        self.stake = stake_training_data
    
    def scale_data(self, train_data, val_data):
        """
        This method scales the data by removing the mean and scaling to unit variance.
        If features_not_to_scale is not empty, those features are not scaled.
        """
        if len(self.features_not_to_scale) == 0:
            self.scaler.fit(train_data)
            train_scaled = self.scaler.transform(train_data)
            val_scaled = self.scaler.transform(val_data)
        
        else:
            categorical_features_train = train_data.loc[:, self.features_not_to_scale]
            continous_features_train = train_data.drop(labels=self.features_not_to_scale, axis=1)
            categorical_features_val = val_data.loc[:, self.features_not_to_scale]
            continous_features_val = val_data.drop(labels=self.features_not_to_scale, axis=1)

            self.scaler.fit(continous_features_train)
            continous_train_scaled = self.scaler.transform(continous_features_train)
            continous_val_scaled = self.scaler.transform(continous_features_val)

            # Combine categorical and scaled features 
            train_scaled = np.concatenate((continous_train_scaled,
                                           categorical_features_train), axis=1)
            val_scaled = np.concatenate((continous_val_scaled,
                                         categorical_features_val), axis=1)
        return train_scaled, val_scaled
        
    def drop_features(self):
        self.dataset = self.dataset.drop(labels=self.ignored_features, axis=1)
        
    def provide_statistics(self):
        return self.scaler.mean_, self.scaler.var_
    
    def prepare_data(self):
        """
        This function wraps the pre-processing methods and split the data into train
        and validation data.
        :return: Training and val data with dimension [batch, features]
        """
        self.drop_features()
        amount_training_data = round(len(self.dataset)*self.stake)
        train_data = self.dataset.iloc[0:amount_training_data, :]
        val_data = self.dataset.iloc[amount_training_data:, :]
        train_preprocessed, val_preporcessed = self.scale_data(train_data, val_data)
        return train_preprocessed, val_preporcessed
