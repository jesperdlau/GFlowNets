
import torch

# Import scripts
from models.train_tb import train_tb
from models.gflownet_model_tb import GFlowNet
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions import torch_helperfunctions as help

# Import Hyperparameters
from config.config import NAME_OF_RUN, PWD, PWD_WORK, NAME_OF_REWARD

# Hyperparameters
NUM_EPISODES = 20 # Should be 5000 for tfbind8
MINIBATCH_SIZE = 10 # Should be 32 for tfbind8
CHECKPOINT_FREQ = 5 # Should be 50?

HIDDEN_SIZE = 2048
N_HIDDEN_LAYERS = 2
N_ACTIONS = 4
LEN_SEQUENCE = 8
LEARNING_RATE = 10**-5
DELTA = 0.001
BETA = 3
OPT_BETAS = (0.9, 0.999)
LOGZ_LEARNING_RATE = 10**-3
HOT_START = False
VERBOSE = True

# Load path
REWARD_PATH = PWD_WORK + "models/saved_models/tfbind8_reward_model_" + NAME_OF_REWARD + ".pt"

# Save path
MODEL_PATH = PWD_WORK + "models/saved_models/tfbind8_gflow_model_" + NAME_OF_RUN + ".tar"

# Set device
device = help.set_device()

# Load model and optimizer
model = GFlowNet(num_hidden=HIDDEN_SIZE, 
                 n_hidden_layers=N_HIDDEN_LAYERS, 
                 n_actions=N_ACTIONS, 
                 len_sequence=LEN_SEQUENCE, 
                 delta=DELTA)

parameters_to_optimize = []
for name, param in model.named_parameters():
    if name != 'logZ':
        parameters_to_optimize.append(param)

logz_optimizer = torch.optim.Adam(params=[model.logZ], lr=LOGZ_LEARNING_RATE)
optimizer = torch.optim.Adam(params=parameters_to_optimize, lr=LEARNING_RATE, betas=OPT_BETAS)

# Load reward function
reward_func = TFBindReward1HOT()

# Train model and save checkpoint to PATH
train_tb(model = model, 
         optimizer = optimizer, 
         logz_optimizer = logz_optimizer,
         reward_func = reward_func, 
         minibatch_size = MINIBATCH_SIZE,
         num_episodes = NUM_EPISODES, 
         checkpoint_freq = CHECKPOINT_FREQ, 
         delta = DELTA, 
         beta = BETA, 
         model_path = MODEL_PATH, 
         reward_path = REWARD_PATH,
         device = device, 
         hot_start = HOT_START, 
         verbose = VERBOSE)




