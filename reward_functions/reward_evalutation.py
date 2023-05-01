
import torch
import torch.nn as nn
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tf_bind_reward import TFBindReward


DATA_FOLDER = "GFlowNets/data/"

# Rasmus path
DATA_FOLDER = "data/"

SEED = 42
TRAIN_SIZE = 4/5
BATCH_SIZE = 100

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"\nUsing {device} device")


X = np.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_8-x.npy")
y = np.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_8-y.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, shuffle=True, random_state=SEED)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

def evaluate_model(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    print(f"Mean squared error: {test_loss / size}")


def plot_fit_obs(model, X, y):
    
    model.eval()
    # dataloader.batch_size = len(dataloader.dataset)
    # s = [(model(X), y) for X, y in dataloader]

    # preds = []
    # labels = []

    with torch.no_grad():
        preds, labels = model(X), y
        # for X, y in dataloader:
        #     pred = model(X)
        #     labels.append(y)
        #     preds.append(pred)

    # preds = np.array(preds)
    # labels = np.array(labels)
    # torch.Tensor.ndim = property(lambda self: len(self.shape))

    plt.scatter(preds.numpy(),labels.numpy())
    plt.xlabel("Predictions")
    plt.ylabel("Labels")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title("Plot of the fitted and observed values")
    plt.show()

if __name__ == "__main__":
    loss = nn.MSELoss()
    model_name = "TFBind_testmodel.pth"

    model = TFBindReward()
    model.load_state_dict(torch.load(model_name))
    
    # trainSet = SequenceDataset(X_train,y_train)
    testSet = TensorDataset(X_test,y_test)

    # BATCH_SIZE = len(y_test)
    # train_dataLoader = DataLoader(trainSet,batch_size=BATCH_SIZE,shuffle=True)
    test_dataLoader =  DataLoader(testSet,batch_size=BATCH_SIZE,shuffle=True)

    evaluate_model(test_dataLoader,model,loss)
    plot_fit_obs(model, X_test,y_test)
