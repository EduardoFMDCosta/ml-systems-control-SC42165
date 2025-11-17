import torch.nn as nn
import torch.nn.functional as F

class NeuralNetworkDynamics(nn.Module):  # inherit the nn.Module class for backpropagation and training functionalities
    def __init__(self):
        super(NeuralNetworkDynamics, self).__init__()
        # TODO: Implement a NN with at least 3 layers, and ReLU activation
        # TODO: Feel free to use any kind of structure (e.g. dense feedforward, residual, etc). This is an engineering choice
        # TODO: You can consult, for instance: https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

        self.fc_in = nn.Linear(5, 10) # input layer: 5 -> 10

        # Residual block
        self.res_fc1 = nn.Linear(10, 10)
        self.res_fc2 = nn.Linear(10, 10)

        self.fc_out = nn.Linear(10, 3) # output layer: 10 -> 3

    def forward(self, x):

        x = F.relu(self.fc_in(x))

        # Residual block
        residual = x
        out = F.relu(self.res_fc1(x))
        out = self.res_fc2(out)
        x = F.relu(out + residual)  # add skip connection

        # Output projection
        x = self.fc_out(x)
        return x