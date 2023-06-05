
import torch

def MinMaxScaler(tensor, min_,max_):
    min_y = torch.min(tensor)
    max_y = torch.max(tensor)
    std = (tensor - min_y) / (max_y - min_y)
    scaled_tensor = std * (max_ - min_) + min_
    return scaled_tensor

def set_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

def train_model(epochs, train_DL,test_DL, model, loss_fn, optimizer,save_as = None,verbose=True):
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