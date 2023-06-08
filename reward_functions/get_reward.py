import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tf_bind_reward_1hot import TFBindReward1HOT
from torch_helperfunctions import train_model, set_device, MinMaxScaler, train_model_earlystopping
from gfp_reward_1hot import GFPReward

def train_tfbind_reward(epochs:int, X_train, y_train, X_test, y_test, save_as = None, patience = 5, verbose = True):
    """
    Trains a TFBindReward1HOT model using the specified training and testing sets for a specified amount of epochs.
    
    :param epochs: The number of times to iterate over the entire training dataset.
    :type epochs: int
    :param X_train: The training input data.
    :type X_train: torch.Tensor
    :param y_train: The training target data.
    :type y_train: torch.Tensor
    :param X_test: The testing input data.
    :type X_test: torch.Tensor
    :param y_test: The testing target data.
    :type y_test: torch.Tensor
    :param save_as: The name to save the trained model as. Defaults to None.
    :type save_as: str
    """    

    device = set_device()
    model = TFBindReward1HOT()
    loss = nn.MSELoss()
    LEARNING_RATE = 0.001
    BATCH_SIZE = 75

    opt = optim.Adam(model.parameters(), LEARNING_RATE)
    
    trainSet = TensorDataset(X_train,y_train)
    testSet = TensorDataset(X_test,y_test)
    
    train_dataLoader = DataLoader(trainSet,batch_size=BATCH_SIZE,shuffle=True)
    test_dataLoader =  DataLoader(testSet,batch_size=BATCH_SIZE,shuffle=True)
    
    train_model_earlystopping(epochs,train_dataLoader,test_dataLoader,model,loss,opt,save_as = save_as, patience=patience, verbose=verbose)

def train_gfp_reward(epochs, X_train, y_train, X_test, y_test, save_as = None, patience = 5, verbose = True):
    """
    Trains a GFPReward model for a given number of epochs using the provided training and testing data. 

    :param epochs: An integer representing the number of times the model should iterate over the training data.
    :param X_train: A tensor of shape (n_samples, n_features) representing the training data.
    :param y_train: A tensor of shape (n_samples,) representing the target values for the training data.
    :param X_test: A tensor of shape (n_samples, n_features) representing the testing data.
    :param y_test: A tensor of shape (n_samples,) representing the target values for the testing data.
    :param save_as: Optional. If provided, the trained model will be saved with the given filename.
    :param patience: Optional. An integer representing the number of epochs to wait before early stopping if the validation loss does not improve.
    
    :return: None.
    """
    
    device = set_device()
    model = GFPReward()
    loss = nn.MSELoss()
    LEARNING_RATE = 0.001
    BATCH_SIZE = 75

    y_train = MinMaxScaler(y_train,0,1)
    y_test = MinMaxScaler(y_test,0,1)

    opt = optim.Adam(model.parameters(), LEARNING_RATE)
    
    trainSet = TensorDataset(X_train,y_train)
    testSet = TensorDataset(X_test,y_test)
    
    train_dataLoader = DataLoader(trainSet,batch_size=BATCH_SIZE,shuffle=True)
    test_dataLoader =  DataLoader(testSet,batch_size=BATCH_SIZE,shuffle=True)
    
    train_model_earlystopping(epochs,train_dataLoader,test_dataLoader,model,loss,opt,save_as = save_as, patience = patience,verbose=verbose)


if __name__ == "__main__":
    # An example of use

    DATA_FOLDER = "data/"
    X_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt")
    y_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_train.pt")
    X_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_test.pt")
    y_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_test.pt")

    SAVE_PATH_AND_NAME = "models/saved_models/tfbind_reward_earlystopping"

    train_tfbind_reward(20,X_train,y_train,X_test,y_test,save_as = SAVE_PATH_AND_NAME,patience=5)