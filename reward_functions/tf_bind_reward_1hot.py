import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



class TFBindReward1HOT(nn.Module):

    def __init__(self, n_hid = 2048, n_hidden_layers = 2):
        super(TFBindReward1HOT, self).__init__()
        n_actions    = 4
        len_sequence = 8
        len_onehot   = n_actions * len_sequence
        
        input_layer   = nn.Linear(len_onehot, n_hid)
        output_layer  = nn.Linear(n_hid, 1)
        act_func      = nn.ReLU()
        
        hidden_layers = []
        for _ in range(n_hidden_layers):
            hidden_layers.append(nn.Linear(n_hid, n_hid))
            hidden_layers.append(act_func)

        model_architecture = [input_layer, act_func, *hidden_layers, output_layer]
        self.model = nn.Sequential(*model_architecture)

    def forward(self,x):
        return self.model(x)

if __name__ == "__main__":
    from torch_helperfunctions import train_model, set_device
    DATA_FOLDER = "GFlowNets/data/"

    # Rasmus path
    DATA_FOLDER = "data/"

    SEED = 42
    TRAIN_SIZE = 4/5
    EPOCHS = 30
    BATCH_SIZE = 100
    LEARNING_RATE = 0.001

    device = set_device()
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

    train_model(EPOCHS,train_dataLoader,test_dataLoader,model,loss,opt,save_as = "models/saved_models/TFBind_1hot_test")
    print("done")
