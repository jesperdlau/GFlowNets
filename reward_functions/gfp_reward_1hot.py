import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from train_reward_function import train_model, set_device

class GFPReward(nn.Module):

    def __init__(self):
        super(GFPReward, self).__init__()

        self.model = nn.Sequential(
                nn.Linear(4740, 1000),
                nn.ReLU(),
                nn.Linear(1000,500),
                nn.ReLU(),
                nn.Linear(500, 250),
                nn.ReLU(),
                nn.Linear(250, 1))
        
    def forward(self,x):
        return self.model(x)
        

if __name__ == "__main__":
    DATA_FOLDER = "GFlowNets/data/"

    # Rasmus
    DATA_FOLDER = "data/"
    TRAIN_SIZE = 4/5
    EPOCHS = 10
    BATCH_SIZE = 75
    LEARNING_RATE = 0.001

    device = set_device()

    print(f"\nUsing {device} device")

    X_train = torch.load(DATA_FOLDER + "gfp/gfp_1hot_X_train.pt")
    y_train = torch.load(DATA_FOLDER + "gfp/gfp_1hot_y_train.pt")
    X_test  = torch.load(DATA_FOLDER + "gfp/gfp_1hot_X_test.pt")
    y_test  = torch.load(DATA_FOLDER + "gfp/gfp_1hot_y_test.pt")

    model = GFPReward()
    loss = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    trainSet = TensorDataset(X_train,y_train)
    testSet = TensorDataset(X_test,y_test)

    train_dataLoader = DataLoader(trainSet,batch_size=BATCH_SIZE,shuffle=True)
    test_dataLoader =  DataLoader(testSet,batch_size=BATCH_SIZE,shuffle=True)

    train_model(EPOCHS,train_dataLoader,test_dataLoader,model,loss,opt,save_as = "GFP_1hot")

