import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from data_handler import get_data

def train(net: nn.Module,
          num_epochs: int,
          batch_size: int,
          lr: float,
          weight_decay: float,
          **kwargs):

    torch.manual_seed(10)

    #Read CSV file containing data
    data = get_data('data.csv')

    train_set = torch.tensor(np.array(data['x'].tolist()), dtype=torch.float32)
    test_set = torch.tensor(np.array(data['y'].tolist()), dtype=torch.float32)

    #Split dataset into train and test set
    x_train, x_test, y_train, y_test = train_test_split(train_set, test_set, test_size = 0.2, random_state = 0)

    # Define loss function
    loss_func = nn.MSELoss()

    # Define optimizer
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay=weight_decay)

    # Training parameters
    batches = int(np.floor(len(y_train) / batch_size))  # how many batches are there when dividing the whole data set

    # Set network to training mode
    net.train()

    Index = np.arange(len(y_train))  # Index, so we can randomly shuffle inputs and outputs

    for epoch in range(num_epochs):
        np.random.shuffle(Index)  # shuffle indices, so batches have randomly selected samples

        loss_epoch = 0.

        for batch in range(batches):
            Index_batch = Index[batch * batch_size:(batch + 1) * batch_size]
            x_batch = x_train[Index_batch]
            y_batch = y_train[Index_batch]

            # Training
            optimizer.zero_grad()
            y_batch_pred = net(x_batch)
            loss = loss_func(y_batch_pred, y_batch)
            loss.backward()
            optimizer.step()

            loss_epoch += loss

        loss_epoch /= batches
        print('Loss for epoch {}/{}: {:0.4e}'.format(epoch, num_epochs, loss_epoch))

    # Save trained model
    torch.save(net.state_dict(), 'model_weights.pth')

    # Evaluate performance in test set
    net.eval()
    with torch.no_grad():
        y_test_pred = net(x_test)
        loss = loss_func(y_test_pred, y_test)
        print(f'Loss for test set: {loss}')

    return net