import torch.nn as nn
import torch.nn.functional as F

class NeuralNetworkDynamics(nn.Module):  # inherit the nn.Module class for backpropagation and training functionalities
    def __init__(self):
        super(NeuralNetworkDynamics, self).__init__()
        # TODO: Implement a Neural Network with at least 3 layers, and ReLU activation
        # TODO: You can consult, for instance: https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
        self.fc1 = nn.Linear(5, 10, bias=True)  # fully connected layer from 3 to 10 dimensions
        self.fc2 = nn.Linear(10, 3, bias=True)  # fully connected layer from 10 to 3 dimensions

    def forward(self, x):
        x = self.fc1(x)  # apply the first fully connected layer, x now has shape n x 10
        x = F.relu(x)  # apply a ReLU activation to the hidden layer
        x = self.fc2(x)  # apply the second fully connected layer, x now has shape n x 3
        return x