
import torch
import torch.nn as nn
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tf_bind_reward import TFBindReward
from tf_bind_reward_1hot import TFBindReward1HOT

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

X_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt")
y_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_train.pt")
X_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_test.pt")
y_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_test.pt")


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
    
    model = TFBindReward1HOT()
    model_name = "TFBind_1hot_test.pth"
    
    # model = TFBindReward()
    # model_name = "TFBind_testmodel.pth"

    model.load_state_dict(torch.load(model_name))
    testSet = TensorDataset(X_test,y_test)

    # train_dataLoader = DataLoader(trainSet,batch_size=BATCH_SIZE,shuffle=True)
    test_dataLoader =  DataLoader(testSet,batch_size=BATCH_SIZE,shuffle=True)

    evaluate_model(test_dataLoader,model,loss)
    plot_fit_obs(model, X_test,y_test)
