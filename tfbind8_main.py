
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
NUM_EPISODES = 1000
UPDATE_FREQ = 10
PATH = "models/saved_models/test_model_1.tar"
PLOT_PATH = "models/test_loss_plot_1.png"

# Set device
device = help.set_device()

# Load model and optimizer
model = GFlowNet(HIDDEN_SIZE)
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

# Load reward function
reward_func = TFBindReward1HOT()
reward_path = "models/saved_models/TFBind_1hot_test.pth"
reward_func.load_state_dict(torch.load(reward_path))

# Train model and save checkpoint to PATH
train(model, optimizer, reward_func, SEQ_LEN, NUM_EPISODES, UPDATE_FREQ, device, PATH, hot_start=True)

# Plot of loss saved to PLOT_PATH
loss_plot(PATH, save_path=PLOT_PATH)





