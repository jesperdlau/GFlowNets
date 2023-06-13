
import torch
import numpy as np

# Import scripts
from MCMC_sampler import MCMCSequenceSampler
from models.random_sampler import SequenceSampler
from models.train_tb import train_tb
from models.tfbind8_model_tb import GFlowNet
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions import torch_helperfunctions as help
from utilities.loss_plot import loss_plot
from evaluation.evaluation import evaluate_modelsampling
from utilities.transformer import Transformer
from MCMC_light_sampler import MCMCLightSequenceSampler


# Hyperparameters
SEQ_LEN = 8 # Shouldn't need to change
HIDDEN_SIZE = 2048
LEARNING_RATE = 10e-4
NUM_EPISODES = 100
UPDATE_FREQ = 32
MODEL_PATH = "models/saved_models/test_model_3.tar"
REWARD_PATH = "models/saved_models/tfbind_reward_model_1.pt"
PLOT_PATH = "models/test_loss_plot_3.png" # TODO: Organize model into model folder, reward into reward folder and plot into plot folder???
SAMPLE_SIZE = 100
BURNIN = 10
DELTA = 0.001
BETA = 3
LOGZ_LEARNING_RATE = 10^-3

DATA_FOLDER = "data/"

# Set device
device = help.set_device()

# Load model and optimizer
model = GFlowNet()

parameters_to_optimize = []
for name, param in model.named_parameters():
    if name != 'logZ':
        parameters_to_optimize.append(param)

logz_optmizer = torch.optim.Adam([model.logZ], LOGZ_LEARNING_RATE)
optimizer = torch.optim.Adam(parameters_to_optimize, LEARNING_RATE)

# Load reward function
reward_func = TFBindReward1HOT()

# Train model and save checkpoint to PATH
train_tb(model, optimizer, reward_func, NUM_EPISODES, UPDATE_FREQ, DELTA, BETA, MODEL_PATH, REWARD_PATH, device, hot_start=False, verbose=True)

# Plot of loss saved to PLOT_PATH
loss_plot(MODEL_PATH, save_path=PLOT_PATH)

# Sample
# MCMC_sampler = MCMCSequenceSampler()
# random_sampler = SequenceSampler()

# random_samples = Transformer.list_list_int_to_tensor_one_hot(random_sampler.sample(SAMPLE_SIZE))
#MCMC_samples = transformer.list_list_int_to_tensor_one_hot(MCMC_sampler.sample(SAMPLE_SIZE, BURNIN))
Gflow_samples = model.sample(SAMPLE_SIZE)

#predict

# random_reward = reward_func(random_samples)
#MCMC_reward = reward_func(MCMC_samples)
gflow_reward = reward_func(Gflow_samples)

#Evaluate

X_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt")

# random_evaluation = evaluate_modelsampling(X_train,random_samples,random_reward, print_stats = True)
#MCMC_evaluation = evaluate_modelsampling(X_train,MCMC_samples,MCMC_reward, print_stats = True)
gflow_evaluation = evaluate_modelsampling(X_train,Gflow_samples,gflow_reward, print_stats = True)




