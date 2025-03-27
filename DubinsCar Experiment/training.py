import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_handler import get_data
from torch.utils.data import TensorDataset, DataLoader

def train(net: nn.Module,
          num_epochs: int,
          batch_size: int,
          lr: float,
          weight_decay: float,
          **kwargs):

    torch.manual_seed(10)

    #Read CSV file containing data
    data = get_data('data.csv')

    train_X = torch.tensor(np.array(data['x'].tolist()), dtype=torch.float32)
    train_y = torch.tensor(np.array(data['y'].tolist()), dtype=torch.float32)

    dataset = TensorDataset(train_X, train_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss function
    loss_func = nn.MSELoss()

    # Define optimizer
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay=weight_decay)

    # Set network to training mode
    net.train()

    for epoch in range(num_epochs):

        loss_epoch = 0.

        for x_batch, y_batch in dataloader:

            # Training
            optimizer.zero_grad()
            y_batch_pred = net(x_batch)
            loss = loss_func(y_batch_pred, y_batch)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for param in net.parameters():
                    param.clamp_(-1, 1)

            loss_epoch += loss

        loss_epoch /= len(dataloader)
        print('Loss for epoch {}/{}: {:0.4e}'.format(epoch, num_epochs, loss_epoch))

    # Save trained model
    torch.save(net.state_dict(), 'model_weights.pth')

    return net