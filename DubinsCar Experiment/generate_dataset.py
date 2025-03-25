import torch
import math
import pandas as pd
from dynamics import DubinsCarDynamics
from regions import HyperRectangle

# Initialize dynamics
dynamics = DubinsCarDynamics(velocity=2.0,
                             control=1.0,
                             disc_step=0.1)

# Define training hypercube
lower = torch.tensor([-3., -3., 0.])
upper = torch.tensor([3., 3., 2 * math.pi])
hypercube = HyperRectangle(lower, upper)

# Data generation parameters
n_samples = 10000
n_steps = 20


data_pairs = []
states = hypercube.get_random_points(n_samples)

for t in range(n_steps):
    previous_states = states.clone().detach()
    states = dynamics(states)

    for previous_state, state in zip(previous_states, states):
        data_pairs.append([previous_state.tolist(), state.tolist()])


df = pd.DataFrame(data_pairs)
df.columns = ['x', 'y']


# Serialize the arrays into strings
df['x'] = df['x'].apply(lambda x: ' '.join(map(str, x)))
df['y'] = df['y'].apply(lambda x: ' '.join(map(str, x)))

# Save the DataFrame to a CSV file
df.to_csv('data.csv',
          index=False)