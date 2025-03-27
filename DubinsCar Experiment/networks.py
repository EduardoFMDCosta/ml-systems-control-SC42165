import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrize

class NeuralDubinsCar(nn.Module):

    def __init__(self):
        super(NeuralDubinsCar, self).__init__()
        self.fc1 = nn.Linear(3, 128, bias=True)
        self.fc2 = nn.Linear(128, 128, bias=True)
        self.fc3 = nn.Linear(128, 3, bias=True)

    def forward(self, x):

        residual = x

        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)

        x = x + residual
        return x

class NeuralPendulum(nn.Module):

    def __init__(self):
        super(NeuralPendulum, self).__init__()
        self.fc1 = nn.Linear(2, 64, bias=True)
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.fc3 = nn.Linear(64, 2, bias=True)

    def forward(self, x):

        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)

        return x


def get_network(dynamics_type: str):
    if dynamics_type == 'DubinsCar':
        return NeuralDubinsCar()
    elif dynamics_type == 'Pendulum':
        return NeuralPendulum()
    else:
        raise ValueError(f"Unknown network: {dynamics_type}")