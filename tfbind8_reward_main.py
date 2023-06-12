import torch
# from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions.get_reward import train_tfbind_reward

# Hyperparameters
NAME_OF_RUN = "100mc_test"
BATCH_SIZE = 256
LEARNING_RATE = 10**-4
EPOCS = 30
N_HID = 2048
N_HIDDEN_LAYERS = 2
BETAS = (0.9, 0.999)
PATIENCE = 5
VERBOSE = True

# HPC
# X_TRAIN_PATH = "/zhome/2e/b/169155/GFlowNet/tfbind8/data/tf_bind_1hot_X_train.pt"
# Y_TRAIN_PATH = "/zhome/2e/b/169155/GFlowNet/tfbind8/data/tf_bind_1hot_y_train.pt"
# X_TEST_PATH = "/zhome/2e/b/169155/GFlowNet/tfbind8/data/tf_bind_1hot_X_test.pt"
# Y_TEST_PATH = "/zhome/2e/b/169155/GFlowNet/tfbind8/data/tf_bind_1hot_y_test.pt" 
# SAVE_PATH_AND_NAME = "/zhome/2e/b/169155/GFlowNet/tfbind8/reward/tfbind_reward_model_1.pt"

# Load path - Shoud be the same
X_TRAIN_PATH = "data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt"
Y_TRAIN_PATH = "data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_train.pt"
X_TEST_PATH = "data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_test.pt"
Y_TEST_PATH = "data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_test.pt" 

# Save path
REWARD_PATH = "models/saved_models/tfbind8_reward_model_" + NAME_OF_RUN + ".pt"

# Load data
X_train = torch.load(X_TRAIN_PATH)
y_train = torch.load(Y_TRAIN_PATH)
X_test  = torch.load(X_TEST_PATH)
y_test  = torch.load(Y_TEST_PATH)

# Train reward function and save model dict to save path
train_tfbind_reward(epochs=EPOCS,
                    X_train = X_train,
                    y_train = y_train,
                    X_test = X_test,
                    y_test = y_test,
                    batch_size=BATCH_SIZE,
                    n_hid=N_HID,
                    n_hidden_layers=N_HIDDEN_LAYERS,
                    save_as = REWARD_PATH,
                    patience = PATIENCE,
                    verbose=VERBOSE)


