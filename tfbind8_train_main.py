import torch
import numpy as np

# Import scripts
from models.train import train_flow_matching
from models.tfbind8_model import GFlowNet
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions import torch_helperfunctions as help

# Hyperparameters
SEQ_LEN = 8
HIDDEN_SIZE = 2048
N_HIDDEN_LAYERS = 2
LEARNING_RATE = 10e-5
NUM_EPISODES = 100 # Should be 5000
UPDATE_FREQ = 32
VERBOSE = True
HOT_START = False
BETAS = (0.9, 0.999)

# HPC
# MODEL_PATH = "/zhome/2e/b/169155/GFlowNet/tfbind8/model/test_model_2048.tar"
# REWARD_PATH = "/zhome/2e/b/169155/GFlowNet/tfbind8/reward/tfbind_reward_model_1.pt"

# Local
MODEL_PATH = "models/saved_models/test_model_2048.tar"
REWARD_PATH = "models/saved_models/tfbind_reward_model_1.pt"

# Initialize 
device = help.set_device()
model = GFlowNet(HIDDEN_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS)
reward_func = TFBindReward1HOT()

# Train model and save checkpoint to PATH
train_flow_matching(model, 
                    optimizer, 
                    reward_func, 
                    SEQ_LEN, 
                    NUM_EPISODES, 
                    UPDATE_FREQ, 
                    MODEL_PATH, 
                    REWARD_PATH, 
                    device, 
                    hot_start=HOT_START, 
                    verbose=VERBOSE)


