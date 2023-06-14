import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .tf_bind_reward_1hot import TFBindReward1HOT
from .torch_helperfunctions import train_model, set_device, MinMaxScaler, train_model_earlystopping
from .gfp_reward_1hot import GFPReward

def train_tfbind_reward(epochs:int, X_train, y_train, X_test, y_test,batch_size = 75, learning_rate = 0.0001, 
                        n_hid = 2048, n_hidden_layers = 2, betas = (0.9, 0.999), save_as = None, patience = 5, verbose = True):
    """
	Trains a TFBindReward1HOT model using the provided training and testing data. 

	:param epochs: Number of training epochs.
	:param X_train: Input training data.
	:param y_train: Target training data.
	:param X_test: Input testing data.
	:param y_test: Target testing data.
	:param n_hid: Number of hidden units in each hidden layer of the model. Default is 2048.
	:param n_hidden_layers: Number of hidden layers in the model. Default is 2.
	:param save_as: Path to save the trained model. Default is None.
	:param patience: Number of epochs to wait before early stopping. Default is 5.
	:param verbose: Whether to print training and validation loss during training. Default is True.

	:returns: None.
	"""
        
    device = set_device()
    model = TFBindReward1HOT(n_hid = n_hid, n_hidden_layers = n_hidden_layers)
    loss = nn.MSELoss()

    opt = optim.Adam(model.parameters(), lr = learning_rate, betas = betas)
    
    trainSet = TensorDataset(X_train,y_train)
    testSet = TensorDataset(X_test,y_test)
    
    train_dataLoader = DataLoader(trainSet,batch_size=batch_size,shuffle=True)
    test_dataLoader =  DataLoader(testSet,batch_size=batch_size,shuffle=True)
    
    train_model_earlystopping(epochs,train_dataLoader,test_dataLoader,model,loss,opt,device,
                              save_as = save_as, patience=patience, verbose=verbose)

def train_gfp_reward(epochs:int, X_train, y_train, X_test, y_test,batch_size = 75, learning_rate = 0.001, 
                        n_hid = 2048,n_hidden_layers = 2,betas = (0.9, 0.999),save_as = None, patience = 5, verbose = True):
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
    model = GFPReward(n_hid = n_hid, n_hidden_layers = n_hidden_layers)
    loss = nn.MSELoss()

    y_train = MinMaxScaler(y_train,0,1)
    y_test = MinMaxScaler(y_test,0,1)

    opt = optim.Adam(model.parameters(), learning_rate, betas = betas)
    
    trainSet = TensorDataset(X_train,y_train)
    testSet = TensorDataset(X_test,y_test)
    
    train_dataLoader = DataLoader(trainSet,batch_size=batch_size,shuffle=True)
    test_dataLoader =  DataLoader(testSet,batch_size=batch_size,shuffle=True)
    
    train_model_earlystopping(epochs,train_dataLoader,test_dataLoader,model,loss,opt,device,save_as = save_as, patience = patience,verbose=verbose)

if __name__ == "__main__":
    # An example of use

    DATA_FOLDER = "data/"
    X_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt")
    y_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_train.pt")
    X_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_test.pt")
    y_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_test.pt")

    SAVE_PATH_AND_NAME = "models/saved_models/tfbind_reward_earlystopping.pth"

    train_tfbind_reward(30,X_train,y_train,X_test,y_test,save_as = SAVE_PATH_AND_NAME,patience=5)