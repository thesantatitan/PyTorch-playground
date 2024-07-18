import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, hidden_layers, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.hidden_layers = []
        for i in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
        self.fc_last = nn.Linear(hidden_layer_size, output_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = F.relu(self.fc_last(x))
        return x


