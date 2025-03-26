import pandas as pd
from dynamics import Dynamics
from regions import HyperRectangle

def generate_dataset(dynamics: Dynamics,
                     hypercube: HyperRectangle,
                     n_samples: int,
                     n_steps: int):

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
    df.to_csv('data.csv', index=False)