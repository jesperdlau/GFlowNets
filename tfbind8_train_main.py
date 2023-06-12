import torch
import numpy as np

# Import scripts
from models.train import train_flow_matching
from models.tfbind8_model import GFlowNet
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions import torch_helperfunctions as help

# Hyperparameters
NAME_OF_RUN = "100mc_test"
NUM_EPISODES = 100 # Should be 5000
VERBOSE = True

HIDDEN_SIZE = 2048
N_HIDDEN_LAYERS = 2
LEARNING_RATE = 10**-5
UPDATE_FREQ = 32
DELTA = 0.001 # Uniform Policy Coefficient
BETA = 3 # Reward Exponent
OPT_BETAS = (0.9, 0.999)
HOT_START = False

# HPC
# REWARD_PATH = "/zhome/2e/b/169155/GFlowNet/tfbind8/reward/tfbind_reward_model_1.pt"
# MODEL_PATH = "/zhome/2e/b/169155/GFlowNet/tfbind8/model/test_model_2048.tar"

# Load path
REWARD_PATH = "models/saved_models/tfbind8_reward_model_" + NAME_OF_RUN + ".pt"

# Save path
MODEL_PATH = "models/saved_models/tfbind8_gflow_model_" + NAME_OF_RUN + ".tar"

# Initialize 
device = help.set_device()
model = GFlowNet(HIDDEN_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=OPT_BETAS)
reward_func = TFBindReward1HOT()

# Train model and save checkpoint to PATH
train_flow_matching(model=model, 
                    optimizer=optimizer, 
                    reward_func=reward_func, 
                    num_episodes=NUM_EPISODES, 
                    update_freq=UPDATE_FREQ, 
                    delta=DELTA,
                    beta=BETA,
                    model_path=MODEL_PATH, 
                    reward_path=REWARD_PATH, 
                    device=device, 
                    hot_start=HOT_START, 
                    verbose=VERBOSE)


