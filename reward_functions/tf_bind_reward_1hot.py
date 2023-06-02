import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

DATA_FOLDER = "GFlowNets/data/"

# Rasmus path
DATA_FOLDER = "data/"

SEED = 42
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

X_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt")
y_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_train.pt")
X_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_test.pt")
y_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_test.pt")

class TFBindReward1HOT(nn.Module):

    def __init__(self):
        super(TFBindReward1HOT, self).__init__()
        
        self.model = nn.Sequential(
                nn.Linear(32, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 1))
        
    def forward(self,x):
        return self.model(x)
    
def train_model(epochs, train_DL,test_DL, model, loss_fn, optimizer,save_as = None,verbose=True):
    for epoch in range(epochs):
        size = len(train_DL.dataset)

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

if __name__ == "__main__":
    model = TFBindReward1HOT()
    loss = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainSet = TensorDataset(X_train,y_train)
    testSet = TensorDataset(X_test,y_test)

    train_dataLoader = DataLoader(trainSet,batch_size=BATCH_SIZE,shuffle=True)
    test_dataLoader =  DataLoader(testSet,batch_size=BATCH_SIZE,shuffle=True)

    train_model(EPOCHS,train_dataLoader,test_dataLoader,model,loss,opt,save_as = "TFBind_1hot_test")
