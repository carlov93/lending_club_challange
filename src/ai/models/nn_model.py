import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, batch_size, input_dim, units_fc1, units_fc2, units_fc3, dropout_rate_fc):
        """
        :param batch_size: Number of samples for each batch.
        :param input_dim: Number of dimension of input data.
        :param units_fc1 - units_fc4: Number of units for each layer
        :param dropout_rate_fc: Percentage of dropout rate for every FC layer.
        """
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.dropout = nn.Dropout(p=dropout_rate_fc)
        self.fc1 = nn.Linear(input_dim, units_fc1)  
        self.fc2 = nn.Linear(units_fc1, units_fc2)
        self.fc3 = nn.Linear(units_fc2, units_fc3)
        self.fc4 = nn.Linear(units_fc3, 2)

    def forward(self, input_data):
        """
        This method defines the forward pass through neural network.
        :param input_data: Input data with dimension [batch, sequence length, features].
        :return: Prediction of class.
        """
        x = self.fc1(input_data)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x
