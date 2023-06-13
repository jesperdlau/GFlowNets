
import torch
import numpy as np

# Import scripts
from models.train_tb import train_tb
from models.tfbind8_model_tb import GFlowNet
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions import torch_helperfunctions as help


# Hyperparameters
NAME_OF_RUN = "100tb_test"

HIDDEN_SIZE = 2048
LEARNING_RATE = 10**-5
NUM_EPISODES = 100 # Should be 5000 for tfbind8
UPDATE_FREQ = 32
DELTA = 0.001
BETA = 3
OPT_BETAS = (0.9, 0.999)
LOGZ_LEARNING_RATE = 10**-3
HOT_START = False
VERBOSE = True

# Load path
REWARD_PATH = "models/saved_models/tfbind8_reward_model_" + NAME_OF_RUN + ".pt"

# Save path
MODEL_PATH = "models/saved_models/tfbind8_gflow_model_" + NAME_OF_RUN + ".tar"

# Set device
device = help.set_device()

# Load model and optimizer
model = GFlowNet()

parameters_to_optimize = []
for name, param in model.named_parameters():
    if name != 'logZ':
        parameters_to_optimize.append(param)

logz_optmizer = torch.optim.Adam(params=[model.logZ], lr=LOGZ_LEARNING_RATE)
optimizer = torch.optim.Adam(params=parameters_to_optimize, lr=LEARNING_RATE, betas=OPT_BETAS)

# Load reward function
reward_func = TFBindReward1HOT()

# Train model and save checkpoint to PATH
train_tb(model = model, 
         optimizer = optimizer, 
         logz_optimizer = logz_optimizer,
         reward_func = reward_func, 
         num_episodes = NUM_EPISODES, 
         update_freq = UPDATE_FREQ, 
         delta = DELTA, 
         beta = BETA, 
         model_path = MODEL_PATH, 
         reward_path = REWARD_PATH,
         device = device, 
         hot_start = HOT_START, 
         verbose = VERBOSE)




