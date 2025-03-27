import torch
from networks import NeuralDubinsCar, NeuralPendulum
from dynamics import DubinsCarDynamics, PendulumDynamics
from plotting import plot_actual_and_predictions, plot_vector_field
from utils import compute_spectral_norm
import pprint

#net = NeuralDubinsCar()
net = NeuralPendulum()
net.load_state_dict(torch.load('model_weights.pth'))
net.eval()

#TODO: Make parameters global (guaranteed to be equal to dataset generator)
# dynamics = DubinsCarDynamics(controller=1,
#                              velocity=2.0,
#                              disc_step=0.1)
dynamics = PendulumDynamics(0.9, 1.0, 1.0, 0.8, 0.05)

#initial_state = torch.tensor([[-1.5, -0.9, 0.2]])
initial_state = torch.tensor([[-0.8, 0.9]])
plot_actual_and_predictions(dynamics=dynamics,
                            net=net,
                            initial_state=initial_state)

plot_vector_field(dynamics=dynamics,
                      net=net)

# Compute spectral norm of linear transformations
spectral_norms = compute_spectral_norm(net)
pprint.pprint(spectral_norms)

# Load the model state_dict
state_dict = torch.load('model_weights.pth')

# Convert weights to a readable format and save to a text file
with open('model_weights_pendulum.txt', 'w') as f:
    for key, value in state_dict.items():
        f.write(f"{key}: {value.numpy()}\n")