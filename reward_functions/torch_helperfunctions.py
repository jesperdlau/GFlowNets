
import torch
from torch.utils.data import DataLoader, TensorDataset

def MinMaxScaler(tensor, min_:int,max_:int):
    min_y = torch.min(tensor)
    max_y = torch.max(tensor)
    std = (tensor - min_y) / (max_y - min_y)
    scaled_tensor = std * (max_ - min_) + min_
    return scaled_tensor

def set_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")
    print(f"\nUsing {device} device")
    return device

def train_model(epochs:int, train_DL:DataLoader,test_DL:DataLoader, model, loss_fn, optimizer,save_as = None,verbose=True):
    for epoch in range(epochs):
        size = len(train_DL.dataset)
        print(f"epoch number: {epoch}")

        #model trainging one epoch
        model.train()
        for batch, (X, y) in enumerate(train_DL):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 25 == 0 and verbose == True:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        # Test set evaluation
        model.eval()
        size = len(test_DL.dataset)
        num_batches = len(test_DL)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in test_DL:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        print(f"Mean squared error: {test_loss / size}\n")

    if save_as:
        torch.save(model.state_dict(), save_as + ".pth")


def train_model_earlystopping(epochs:int, train_DL:DataLoader, test_DL:DataLoader, model, loss_fn, optimizer, save_as=None, verbose=True, patience=5):
    best_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        size = len(train_DL.dataset)
        print(f"epoch number: {epoch}")

        # Model training one epoch
        model.train()
        for batch, (X, y) in enumerate(train_DL):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 25 == 0 and verbose:
                loss_val, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
        
        # Test set evaluation
        model.eval()
        size = len(test_DL.dataset)
        num_batches = len(test_DL)
        test_loss = 0

        with torch.no_grad():
            for X, y in test_DL:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()

        mean_squared_error = test_loss / size
        print(f"Mean squared error: {mean_squared_error}\n")

        # Check for early stopping
        if mean_squared_error < best_loss:
            best_loss = mean_squared_error
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Validation loss hasn't improved for {counter} epochs. Stopping early.")
                print(f"Model saved after {epoch} epochs")
                break

    if save_as:
        torch.save(best_model_state, save_as + ".pth")
    else:
        model.load_state_dict(best_model_state)