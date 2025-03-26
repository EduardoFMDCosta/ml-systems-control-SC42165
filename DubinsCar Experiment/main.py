import torch
from regions import HyperRectangle
from dynamics import get_dynamics
from data_handler import parse_arguments, load_params
from generate_dataset import generate_dataset
from networks import get_network
from training import train
from utils import compute_spectral_norm
from plotting import plot_actual_and_predictions
import pprint

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type = "Pendulum",
        num_samples = 100000,
        num_steps = 1,
        regenerate_set = True,
        num_epochs = 10,
        batch_size = 32,
        lr = 0.001,
        weight_decay = 0.00001
    )
    params = load_params(args)

    # Initiate dynamics
    dynamics = get_dynamics(**params)

    # Build training space
    lower = torch.tensor(params["lower"])
    upper = torch.tensor(params["upper"])
    hypercube = HyperRectangle(lower, upper)

    if args.regenerate_set:
        generate_dataset(dynamics, hypercube, args.num_samples, args.num_steps)

    net = get_network(args.dynamics_type)
    net = train(net, **params)

    initial_state = torch.tensor([[-1.5, -0.9]])
    plot_actual_and_predictions(dynamics=dynamics, net=net, initial_state=initial_state)

    # Compute spectral norm of linear transformations
    spectral_norms = compute_spectral_norm(net)
    pprint.pprint(spectral_norms)