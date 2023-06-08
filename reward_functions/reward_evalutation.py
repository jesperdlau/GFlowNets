
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tf_bind_reward_1hot import TFBindReward1HOT
from gfp_reward_1hot import GFPReward
from scipy.stats import pearsonr

from torch_helperfunctions import set_device, MinMaxScaler

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
     
    preds = preds.squeeze().numpy()
    labels = labels.squeeze().numpy()

    # cor = np.corrcoef(preds,labels)
    cor = pearsonr(preds,labels)
    print(cor)

    plt.scatter(preds,labels)
    plt.xlabel("Predictions")
    plt.ylabel("Labels")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title("Plot of the fitted and observed values")
    plt.show()

if __name__ == "__main__":

    # Rasmus path
    DATA_FOLDER = "data/"

    SEED = 42
    TRAIN_SIZE = 4/5
    BATCH_SIZE = 100

    device = set_device()

    # # TFBIND EVALUTAITON
    # X_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_test.pt")
    # y_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_test.pt")    
    # model = TFBindReward1HOT()
    # model_name = "TFBind_1hot_test.pth"

    # GFP EVALUTATION
    X_test  = torch.load(DATA_FOLDER + "gfp/gfp_1hot_X_test.pt")
    y_test  = torch.load(DATA_FOLDER + "gfp/gfp_1hot_y_test.pt")
    y_test = MinMaxScaler(y_test,0,1)
    model = GFPReward()
    model_name = "GFP_1hot.pth"

    loss = nn.MSELoss() 
    model.load_state_dict(torch.load(model_name))
    testSet = TensorDataset(X_test,y_test)

    test_dataLoader =  DataLoader(testSet,batch_size=BATCH_SIZE,shuffle=True)

    evaluate_model(test_dataLoader,model,loss)
    plot_fit_obs(model, X_test,y_test)