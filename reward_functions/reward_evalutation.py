
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tf_bind_reward_1hot import TFBindReward1HOT
from gfp_reward_1hot import GFPReward
from scipy.stats import pearsonr

from torch_helperfunctions import set_device, MinMaxScaler

def evaluate_model(X_test,y_test, model, MinMaxScale = False, verbose = False):
    if MinMaxScale:
        y_test = MinMaxScaler(y_test)

    testSet = TensorDataset(X_test,y_test)
    test_dataLoader =  DataLoader(testSet,batch_size=BATCH_SIZE,shuffle=True)
    size = len(test_dataLoader.dataset)
    num_batches = len(test_dataLoader)
    test_loss, correct = 0, 0
    loss_fn = nn.MSELoss() 

    with torch.no_grad():
        for X, y in test_dataLoader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    if verbose:
        print(f"Mean squared error: {test_loss / size}")

    return test_loss / size

    


def plot_fit_obs(X, y, model, save_path, MinMax = False):
    
    model.eval()

    with torch.no_grad():
        preds, labels = model(X), y
     
    if MinMax:
        labels = MinMaxScaler(labels)

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
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":

    # Rasmus path
    DATA_FOLDER = "data/"

    SEED = 42
    TRAIN_SIZE = 4/5
    BATCH_SIZE = 100

    device = set_device()

    # # TFBIND EVALUTAITON
    X_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_test.pt")
    y_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_test.pt")    
    model = TFBindReward1HOT()
    model_name = "models/saved_models/tfbind_reward_earlystopping.pth"

    # GFP EVALUTATION
    # X_test  = torch.load(DATA_FOLDER + "gfp/gfp_1hot_X_test.pt")
    # y_test  = torch.load(DATA_FOLDER + "gfp/gfp_1hot_y_test.pt")
    # y_test = MinMaxScaler(y_test,0,1)
    # model = GFPReward()
    # model_name = "GFP_1hot.pth"

    model.load_state_dict(torch.load(model_name))

    evaluate_model(X_test,y_test,model)
    plot_fit_obs(X_test,y_test, model, "plots/plot_fit_obs.png")