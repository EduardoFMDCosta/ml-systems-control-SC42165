import matplotlib.pyplot as plt
import torch
from dynamics import Dynamics
import torch.nn as nn

plt.style.use('seaborn-v0_8-bright')

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def plot_actual_and_predictions(dynamics: Dynamics,
                                net: nn.Module,
                                initial_state: torch.Tensor):

    state_actual = initial_state
    state_nn = initial_state.clone().detach()

    plt.scatter(state_actual[0][0], state_actual[0][1], color='blue')
    plt.text(state_actual[0][0] + 0.05, state_actual[0][1], '0', fontsize=10, color='blue', weight='bold')

    for t in range(1, 10):
        # True dynamics
        state_actual = dynamics(state_actual)
        plt.scatter(state_actual[0][0], state_actual[0][1], color='blue')
        plt.text(state_actual[0][0] + 0.05, state_actual[0][1], t, fontsize=10, color='blue', weight='bold')

        # NN dynamics
        with torch.no_grad():
            state_nn = net(state_nn)

        plt.scatter(state_nn[0][0], state_nn[0][1], color='red')
        plt.text(state_nn[0][0] + 0.05, state_nn[0][1], t, fontsize=10, color='red', weight='bold')

    plt.text(0.05, 0.95, '• Actual dynamics', transform=plt.gca().transAxes, color='blue', verticalalignment='top')
    plt.text(0.05, 0.87, '• NN prediction', transform=plt.gca().transAxes, color='red', verticalalignment='top')

    plt.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)
    plt.show()