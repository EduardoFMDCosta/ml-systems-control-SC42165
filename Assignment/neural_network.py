import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Neural_Network_Dynamics(nn.Module):  # inherit the nn.Module class for backpropagation and training functionalities
    def __init__(self):
        super(Neural_Network_Dynamics, self).__init__()
        self.fc1 = nn.Linear(5, 10, bias=True)  # fully connected layer from 3 to 10 dimensions
        self.fc2 = nn.Linear(10, 3, bias=True)  # fully connected layer from 10 to 3 dimensions

    def forward(self, x):
        x = self.fc1(x)  # apply the first fully connected layer, x now has shape n x 10
        x = F.relu(x)  # apply a ReLU activation to the hidden layer
        x = self.fc2(x)  # apply the second fully connected layer, x now has shape n x 3
        return x