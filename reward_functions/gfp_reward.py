import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
# import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

DATA_FOLDER = "GFlowNets/data/"
#DATA_FOLDER = "./data/"

# Rasmus
DATA_FOLDER = "data/"
TRAIN_SIZE = 4/5
EPOCHS = 30
BATCH_SIZE = 100
LEARNING_RATE = 0.001


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"\nUsing {device} device")

SEED = 42

X = np.load(DATA_FOLDER + "gfp/gfp-x.npy")
y = np.load(DATA_FOLDER + "gfp/gfp-y.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, shuffle=True, random_state=SEED)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


class GFPReward(nn.Module):

    def __init__(self):
        super(GFPReward, self).__init__()

        self.model = nn.Sequential(
                nn.Linear(237, 400),
                nn.ReLU(),
                nn.Linear(400,200),
                nn.ReLU(),
                nn.Linear(200, 50),
                nn.ReLU(),
                nn.Linear(50, 1))
        
    def forward(self,x):
        return self.model(x)
        
# inspired by pytorch documentation: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def inner_train(dataloader, model, loss_fn, optimizer, verbose = True):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
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

# Inspired by pytorch documentation: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            test_loss += loss_fn(pred, y).item()
            
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    print(f"Mean squared error: {test_loss / size}")


def train_model(epochs, train_DL,test_DL, model, loss_fn, optimizer,save_as = None):
    for epoch in range(epochs):
        print(f"\nEpoch number: {epoch}")
        inner_train(train_DL,model,loss_fn,optimizer)
        test_loop(test_DL,model,loss_fn)

    if save_as:
        torch.save(model.state_dict(), save_as + ".pth")

if __name__ == "__main__":
    model = GFPReward()
    loss = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    trainSet = TensorDataset(X_train,y_train)
    testSet = TensorDataset(X_test,y_test)

    train_dataLoader = DataLoader(trainSet,batch_size=BATCH_SIZE,shuffle=True)
    test_dataLoader =  DataLoader(testSet,batch_size=BATCH_SIZE,shuffle=True)

    train_model(EPOCHS,train_dataLoader,test_dataLoader,model,loss,opt,save_as = "GFP_testmodel")

