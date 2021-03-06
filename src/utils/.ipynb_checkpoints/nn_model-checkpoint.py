import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

class AnalysisLayer(nn.Module):
    def __init__(self):
        super(AnalysisLayer, self).__init__()
    
    def forward(self, x):
        global latent_space
        latent_space = x.detach()
        return x

class LstmMse(nn.Module):
    def __init__(self, batch_size, input_dim, n_hidden_lstm, n_layers, dropout_rate, n_hidden_fc):
        super(LstmMse, self).__init__()
        # Attributes for LSTM Network
        self.input_dim = input_dim
        self.n_hidden_lstm = n_hidden_lstm
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.n_hidden_fc = n_hidden_fc
        
        # Definition of NN layer
        # batch_first = True because dataloader creates batches and batch_size is 0. dimension
        self.lstm = nn.LSTM(input_size = self.input_dim, 
                            hidden_size = self.n_hidden_lstm, 
                            num_layers = self.n_layers, 
                            batch_first = True, 
                            dropout = self.dropout_rate)
        self.fc1 = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(self.n_hidden_fc, self.input_dim)
        
    def forward(self, input_data, hidden):
        # Forward propagate LSTM
        # LSTM in Pytorch return two results: the first one usually called output 
        # and the second one (hidden_state, cell_state). 
        lstm_out, (hidden_state, cell_state) = self.lstm(input_data, hidden)

        # LSTM returns as output all the hidden_states for all the timesteps (seq), 
        # in other words all of the hidden states throughout
        # the sequence.
        # Thus we have to select the output from the last sequence (last hidden state of sequence)
        # Length of input data can varry 
        length_seq = input_data.size()[1]
        last_out = lstm_out[:,length_seq-1,:]
        
        # Forward path through the subsequent fully connected tanh activation neural network
        out_y_hat = self.fc1(last_out)
        out_y_hat = self.dropout(out_y_hat)
        out_y_hat = F.tanh(out_y_hat)
        out_y_hat = self.fc2(out_y_hat)
        return out_y_hat

    def init_hidden(self):
        # This method is for initializing hidden state as well as cell state
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        h0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        c0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        return [t for t in (h0, c0)]

class LstmMle(nn.Module):
    def __init__(self, batch_size, input_dim, n_hidden_lstm, n_layers, dropout_rate, n_hidden_fc):
        super(LstmMle, self).__init__()
        # Attributes for LSTM Network
        self.input_dim = input_dim
        self.n_hidden_lstm = n_hidden_lstm
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.n_hidden_fc = n_hidden_fc
        
        # Definition of NN layer
        # batch_first = True because dataloader creates batches and batch_size is 0. dimension
        self.lstm = nn.LSTM(input_size = self.input_dim, 
                            hidden_size = self.n_hidden_lstm, 
                            num_layers = self.n_layers, 
                            batch_first = True, 
                            dropout = self.dropout_rate)
        self.fc1 = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc_y_hat = nn.Linear(self.n_hidden_fc, self.input_dim)
        self.fc_tau = nn.Linear(self.n_hidden_fc, self.input_dim)
        
    def forward(self, input_data, hidden):
        # Forward propagate LSTM
        # LSTM in Pytorch return two results: the first one usually called output 
        # and the second one (hidden_state, cell_state). 
        lstm_out, (hidden_state, cell_state) = self.lstm(input_data, hidden)

        # LSTM returns as output all the hidden_states for all the timesteps (seq), 
        # in other words all of the hidden states throughout
        # the sequence.
        # Thus we have to select the output from the last sequence (last hidden state of sequence)
        # Length of input data can varry 
        length_seq = input_data.size()[1]
        last_out = lstm_out[:,length_seq-1,:]

        # Forward path through the subsequent fully connected tanh activation 
        # neural network with 2q output channels
        out = self.fc1(last_out)
        out = self.dropout(out)
        out = F.tanh(out)
        y_hat = self.fc_y_hat(out)
        tau = self.fc_tau(out)
        return [y_hat, tau]
    
    def init_hidden(self):
        # This method is for initializing hidden state as well as cell state
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        h0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        c0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        return [t for t in (h0, c0)]
    
class LstmMultiTaskLearning(nn.Module):
    def __init__(self, batch_size, input_dim, n_hidden_lstm, n_layers, 
                 dropout_rate, n_hidden_fc_prediction, n_hidden_fc_ls_analysis):
        super(LstmMultiTaskLearning, self).__init__()
        # Attributes for LSTM Network
        self.input_dim = input_dim
        self.n_hidden_lstm = n_hidden_lstm
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.n_hidden_fc_prediction = n_hidden_fc_prediction
        self.n_hidden_fc_ls_analysis = n_hidden_fc_ls_analysis
        self.current_latent_space = None
        
        # define strcture of model
        self.sharedlayer = nn.LSTM(input_size = self.input_dim, 
                                   hidden_size = self.n_hidden_lstm, 
                                   num_layers = self.n_layers, 
                                   batch_first = True, 
                                   dropout = self.dropout_rate)
        
        self.prediction_network = nn.Sequential(nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_prediction),
                                                nn.Dropout(p=self.dropout_rate),
                                                nn.Tanh(),
                                                nn.Linear(self.n_hidden_fc_prediction, self.input_dim)
                                               )
        
        self.latent_space_analyse_network = nn.Sequential(nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_ls_analysis),
                                                          nn.Dropout(p=self.dropout_rate),
                                                          nn.Tanh(),
                                                          AnalysisLayer(),
                                                          nn.Linear(self.n_hidden_fc_ls_analysis, self.input_dim)
                                                          )        

    def forward(self, input_data, hidden):
        # Forward propagate LSTM
        # LSTM in Pytorch return two results: the first one usually called output 
        # and the second one (hidden_state, cell_state). 
        lstm_out, (hidden_state, cell_state)= self.sharedlayer(input_data, hidden)
        
        # LSTM returns as output all the hidden_states for all the timesteps (seq), 
        # in other words all of the hidden states throughout the sequence.
        # Thus we have to select the output from the last sequence (last hidden state of sequence).
        # Length of input data can varry 
        length_seq = input_data.size()[1]
        last_out = lstm_out[:,length_seq-1,:]
        
        # Define forward pass through both sub-networks
        prediction = self.prediction_network(last_out)
        _ = self.latent_space_analyse_network(last_out)
        
        # Save latent space
        self.current_latent_space = latent_space
        
        return prediction, _
        
    def init_hidden(self):
        # This method is for initializing hidden state as well as cell state
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        h0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        c0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        return [t for t in (h0, c0)]
    