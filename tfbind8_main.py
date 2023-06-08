
import torch
import numpy as np

# Import scripts
from models.train import train
from models.tfbind8_model import GFlowNet
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions import torch_helperfunctions as help
from utilities.loss_plot import loss_plot


# Hyperparameters
SEQ_LEN = 8 # Shouldn't need to change
HIDDEN_SIZE = 1024
LEARNING_RATE = 3e-4
NUM_EPISODES = 100
UPDATE_FREQ = 10
MODEL_PATH = "models/saved_models/test_model_3.tar"
REWARD_PATH = "models/saved_models/tfbind_reward_earlystopping.pth"
PLOT_PATH = "models/test_loss_plot_3.png" # TODO: Organize model into model folder, reward into reward folder and plot into plot folder???

# Set device
device = help.set_device()

# Load model and optimizer
model = GFlowNet(HIDDEN_SIZE)
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

# Load reward function
reward_func = TFBindReward1HOT()

# Train model and save checkpoint to PATH
train(model, optimizer, reward_func, SEQ_LEN, NUM_EPISODES, UPDATE_FREQ, MODEL_PATH, REWARD_PATH, device, hot_start=False, verbose=True)

# Plot of loss saved to PLOT_PATH
loss_plot(MODEL_PATH, save_path=PLOT_PATH)






