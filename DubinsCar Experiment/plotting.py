import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from dynamics import Dynamics


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

    text_gap = 0.01

    plt.scatter(state_actual[0][0], state_actual[0][1], color='blue')
    plt.text(state_actual[0][0] + text_gap, state_actual[0][1], '0', fontsize=10, color='blue', weight='bold')

    for t in range(1, 250):
        # True dynamics
        state_actual = dynamics(state_actual)
        plt.scatter(state_actual[0][0], state_actual[0][1], color='blue')
        plt.text(state_actual[0][0] + text_gap, state_actual[0][1], t, fontsize=10, color='blue', weight='bold')

        # NN dynamics
        with torch.no_grad():
            state_nn = net(state_nn)

        plt.scatter(state_nn[0][0], state_nn[0][1], color='red')
        plt.text(state_nn[0][0] + text_gap, state_nn[0][1], t, fontsize=10, color='red', weight='bold')

    plt.text(0.05, 0.95, '• Actual dynamics', transform=plt.gca().transAxes, color='blue', verticalalignment='top')
    plt.text(0.05, 0.87, '• NN prediction', transform=plt.gca().transAxes, color='red', verticalalignment='top')

    plt.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)
    plt.show()

def plot_vector_field(dynamics: Dynamics,
                      net: nn.Module):

    # Define grid
    x_min, x_max, y_min, y_max = -1.5, 1.5, -2, 2  # Define plotting range
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)  # Shape (N, 2)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # Compute vector field
    #U, V = dynamics(grid_tensor).T.numpy()  # Extract components

    U, V = net(grid_tensor).detach().T.numpy()  # Extract components

    # Plot vector field
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(X, Y, U.reshape(X.shape), V.reshape(Y.shape), color='b', angles='xy')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True)
    plt.show()