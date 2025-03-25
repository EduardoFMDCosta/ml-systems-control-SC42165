import torch.nn as nn
import torch.nn.functional as F

class NeuralDubinsCar(nn.Module):

    def __init__(self):
        super(NeuralDubinsCar, self).__init__()
        self.fc1 = nn.Linear(3, 16, bias=True)
        self.fc2 = nn.Linear(16, 32, bias=True)
        self.fc3 = nn.Linear(32, 16, bias=True)
        self.fc4 = nn.Linear(16, 3, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        x = self.fc4(x)
        return x