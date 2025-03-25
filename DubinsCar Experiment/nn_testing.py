import torch
from NeuralDubinsCar import NeuralDubinsCar
from dynamics import DubinsCarDynamics
from plotting import plot_actual_and_predictions
from analysis import compute_spectral_norm

net = NeuralDubinsCar()
net.load_state_dict(torch.load('model_weights.pth'))
net.eval()
print(net)

dynamics = DubinsCarDynamics(velocity=2.0, #TODO: Make parameters global (guaranteed to be equal to dataset generator)
                             control=1.0,
                             disc_step=0.1)
initial_state = torch.tensor([[0.0, 1.0, 0.5]])
plot_actual_and_predictions(dynamics=dynamics,
                            net=net,
                            initial_state=initial_state)

# Compute spectral norm of linear transformations
spectral_norms = compute_spectral_norm(net)
print(spectral_norms)

# Load the model state_dict
state_dict = torch.load('model_weights.pth')

# Convert weights to a readable format and save to a text file
with open('model_weights.txt', 'w') as f:
    for key, value in state_dict.items():
        f.write(f"{key}: {value.numpy()}\n")