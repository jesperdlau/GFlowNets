import torch
# from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions.get_reward import train_tfbind_reward

# Import Hyperparameters
import config
NAME_OF_RUN = config.NAME_OF_RUN
PWD = config.PWD
PWD_WORK = config.PWD_WORK

# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 10**-4
EPOCHS = 30
N_HID = 2048
N_HIDDEN_LAYERS = 2
BETAS = (0.9, 0.999)
PATIENCE = 5
VERBOSE = True

# Load path
X_TRAIN_PATH = PWD + "data/tfbind8/tfbind8_X_train.pt"
Y_TRAIN_PATH = PWD + "data/tfbind8/tfbind8_y_train.pt"
X_VALID_PATH = PWD + "data/tfbind8/tfbind8_X_valid.pt"
Y_VALID_PATH = PWD + "data/tfbind8/tfbind8_y_valid.pt"

# Save path
REWARD_PATH = PWD_WORK + "models/saved_models/tfbind8_reward_model_" + NAME_OF_RUN + ".pt"

# Load data
X_train = torch.load(X_TRAIN_PATH)
y_train = torch.load(Y_TRAIN_PATH)
X_valid  = torch.load(X_VALID_PATH)
y_valid  = torch.load(Y_VALID_PATH)

# Train reward function and save model dict to save path
# Note: Validation set is passed to test set
train_tfbind_reward(epochs=EPOCHS,
                    X_train = X_train,
                    y_train = y_train,
                    X_test = X_valid,
                    y_test = y_valid,
                    batch_size=BATCH_SIZE,
                    n_hid=N_HID,
                    n_hidden_layers=N_HIDDEN_LAYERS,
                    save_as = REWARD_PATH,
                    patience = PATIENCE,
                    verbose=VERBOSE)


