import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from train_reward_function import train_model

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

if __name__ == "__main__":
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

    model = TFBindReward1HOT()
    loss = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainSet = TensorDataset(X_train,y_train)
    testSet = TensorDataset(X_test,y_test)

    train_dataLoader = DataLoader(trainSet,batch_size=BATCH_SIZE,shuffle=True)
    test_dataLoader =  DataLoader(testSet,batch_size=BATCH_SIZE,shuffle=True)

    train_model(EPOCHS,train_dataLoader,test_dataLoader,model,loss,opt,save_as = "TFBind_1hot_test")
