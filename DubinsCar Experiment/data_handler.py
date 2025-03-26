import numpy as np
import pandas as pd
import json
import argparse
import os

dir = os.path.dirname(os.path.abspath(__file__))

def get_data(file_name: str):

    data = pd.read_csv(file_name, index_col=False)

    # Deserialize the strings back into NumPy arrays
    data['x'] = data['x'].apply(lambda x: np.array(x.split(), dtype=float))
    data['y'] = data['y'].apply(lambda x: np.array(x.split(), dtype=float))

    return data

def load_json(filename: str):
    file_path = os.path.join(dir, "configs", f"{filename}.json")
    with open(file_path, "r") as read_file:
        data = json.load(read_file)
    return data


def param_handler(param_name: str, dataset_name: str):
    params = load_json(param_name)[dataset_name]
    return params


def parse_arguments(
        dynamics_type: str,
        num_samples: int,
        num_steps: int,
        regenerate_set: bool,
        num_epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float
):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dynamics_type',
                        type=str,
                        choices=['DubinsCar', 'Pendulum'],
                        default=dynamics_type,
                        help='Type of dynamics to use.')
    parser.add_argument('--num_samples',
                        type=int,
                        default=num_samples,
                        help='Number of samples.')
    parser.add_argument('--num_steps',
                        type=int,
                        default=num_steps,
                        help='Number of trajectory steps.')
    parser.add_argument('--regenerate_set',
                        type=bool,
                        default=regenerate_set,
                        help='')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=num_epochs,
                        help='')
    parser.add_argument('--batch_size',
                        type=int,
                        default=batch_size,
                        help='')
    parser.add_argument('--lr',
                        type=float,
                        default=lr,
                        help='')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=weight_decay,
                        help='')

    return parser.parse_args()


def load_params(args):
    dynamics_params = param_handler(
        param_name="configs",
        dataset_name=args.dynamics_type,
    )

    return {"dynamics_type": args.dynamics_type,
            "num_samples": args.num_samples,
            "num_steps": args.num_steps,
            "regenerate_set": args.regenerate_set,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            **dynamics_params}