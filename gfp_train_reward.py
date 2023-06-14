import torch
from reward_functions.get_reward import train_gfp_reward

# Import Hyperparameters
from config.config import NAME_OF_RUN, PWD, PWD_WORK, NAME_OF_REWARD

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
X_TRAIN_PATH = PWD + "data/gfp/gfp_X_train.pt"
Y_TRAIN_PATH = PWD + "data/gfp/gfp_y_train.pt"
X_VALID_PATH = PWD + "data/gfp/gfp_X_test.pt"
Y_VALID_PATH = PWD + "data/gfp/gfp_y_test.pt"

# Save path
REWARD_PATH = PWD_WORK + "models/saved_models/gfp_reward_model_" + NAME_OF_REWARD + ".pt"

# Load data
X_train = torch.load(X_TRAIN_PATH)
y_train = torch.load(Y_TRAIN_PATH)
X_valid  = torch.load(X_VALID_PATH)
y_valid  = torch.load(Y_VALID_PATH)

# Train reward function and save model dict to save path
# Note: Validation set is passed to test set
train_gfp_reward(epochs=EPOCHS,
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


