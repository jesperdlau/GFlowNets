
import torch
import numpy as np
import pandas as pd
import time

# Import scripts
from inference.random_sampler import SequenceSampler
from inference.MCMC_sampler import MCMCSequenceSampler
#from models.tfbind8_model import GFlowNet
from models.tfbind8_model_tb import GFlowNet
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions import torch_helperfunctions as help
from utilities.transformer import Transformer

# Import Hyperparameters
from config.config import NAME_OF_RUN, PWD, PWD_WORK, NAME_OF_REWARD

# Hyperparameters
SAMPLE_SIZE = 100 # Default for tfbind8 is K*t or 128*5 = 640
NUMBER_OF_MCMC_RANDOM = 10
VERBOSE = True

# Hyperparameters for mcmc
BURNIN = 100 # Default 100 grid search
GAMMA_A = 0.5 # Default 0.5 from grid search
GAMMA_SCALE = 0.5 # Default 0.5 from grid search
ALPHABET = ['A', 'C', 'G', 'T']

# Load paths
X_TRAIN_PATH = PWD_WORK + "data/tfbind8/tfbind8_X_train.pt"
MODEL_PATH = PWD_WORK + "models/saved_models/tfbind8_gflow_model_" + NAME_OF_RUN + ".tar"
REWARD_PATH = PWD_WORK + "models/saved_models/tfbind8_reward_model_" + NAME_OF_REWARD + ".pt"

# Save paths
RANDOM_SAMPLES_PATH = PWD_WORK + "inference/tfbind8_random_samples_" + NAME_OF_RUN + ".pt"
RANDOM_REWARDS_PATH = PWD_WORK + "inference/tfbind8_random_rewards_" + NAME_OF_RUN + ".pt"
GFLOW_SAMPLES_PATH = PWD_WORK + "inference/tfbind8_gflow_samples_" + NAME_OF_RUN + ".pt"
GFLOW_REWARDS_PATH = PWD_WORK + "inference/tfbind8_gflow_rewards_" + NAME_OF_RUN + ".pt"
MCMC_SAMPLES_PATH = PWD_WORK + "inference/tfbind8_mcmc_samples_" + NAME_OF_RUN + ".pt"
MCMC_REWARDS_PATH = PWD_WORK + "inference/tfbind8_mcmc_rewards_" + NAME_OF_RUN + ".pt"

# Initialize
device = help.set_device()
start_time = time.time()
now = start_time
X_train = torch.load(X_TRAIN_PATH)
model_dict = torch.load(MODEL_PATH, map_location=device)
models = model_dict["model_state_dict"]

# Setup reward function # TODO: Possibly replace reward function with oracle
reward_func = TFBindReward1HOT()
reward_func.to(device)
reward_func.load_state_dict(torch.load(REWARD_PATH, map_location=device))
reward_func.eval()

# Sample random
random_samples_list, random_rewards_list = [], []
for i in range(NUMBER_OF_MCMC_RANDOM):
    random_sampler = SequenceSampler()
    random_samples = random_sampler.sample_onehot(SAMPLE_SIZE)
    random_samples = random_samples.to(device)
    random_rewards = reward_func(random_samples)
    random_samples_list.append(random_samples.cpu())
    random_rewards_list.append(random_rewards.cpu())

# Convert to torch
random_samples = torch.stack(random_samples_list, dim=0)
random_rewards = torch.stack(random_rewards_list, dim=0)
print("Random sampling complete")

# Sample MCMC
mcmc_samples_list, mcmc_rewards_list = [], []
for i in range(NUMBER_OF_MCMC_RANDOM):
    if VERBOSE: 
        print(f"MCMC#{i+1} / {NUMBER_OF_MCMC_RANDOM} \t Iter time:{time.time() - now:.2f} s \t Time since beginning:{time.time() - start_time:.2f} s")
        now = time.time()
    mcmc_sampler = MCMCSequenceSampler(burnin=BURNIN, a=GAMMA_A, scale=GAMMA_SCALE)
    mcmc_sample_list = mcmc_sampler.sample(SAMPLE_SIZE)
    mcmc_samples = mcmc_samples.to(device)
    mcmc_rewards = reward_func(mcmc_samples)
    mcmc_samples_list.append(mcmc_samples.cpu())
    mcmc_rewards_list.append(mcmc_rewards.cpu())

# Convert to torch
mcmc_samples = torch.stack(mcmc_samples_list, dim=0)
mcmc_rewards = torch.stack(mcmc_rewards_list, dim=0)
print("MCMC sampling complete")

# Setup GFlow
n_hid, n_hidden_layers = model_dict["n_hid"], model_dict["n_hidden_layers"] 
model = GFlowNet(n_hid=n_hid, n_hidden_layers=n_hidden_layers)
model.to(device)

# Sample GFlow
gflow_samples_list, gflow_rewards_list = [], []
for i, model_state in enumerate(models):
    if VERBOSE: 
        print(f"GFlow#{i+1} / {len(models)} \t Iter time:{time.time() - now:.2f} s \t Time since beginning:{time.time() - start_time:.2f} s")
        now = time.time()
    model.load_state_dict(model_state)
    model.eval()
    gflow_batch_samples = model.sample(SAMPLE_SIZE) # Note: This is the most time consuming step
    gflow_batch_samples.to(device)
    gflow_batch_reward = reward_func(gflow_batch_samples)
    gflow_samples_list.append(gflow_batch_samples.cpu())
    gflow_rewards_list.append(gflow_batch_reward.cpu())

# Convert to torch
gflow_samples = torch.stack(gflow_samples_list, dim=0)
gflow_rewards = torch.stack(gflow_rewards_list, dim=0)
print("Gflow sampling complete")

# Save 
torch.save(random_samples, RANDOM_SAMPLES_PATH)
torch.save(random_rewards, RANDOM_REWARDS_PATH)
torch.save(gflow_samples, GFLOW_SAMPLES_PATH)
torch.save(gflow_rewards, GFLOW_REWARDS_PATH)
torch.save(mcmc_samples, MCMC_SAMPLES_PATH)
torch.save(mcmc_rewards, MCMC_REWARDS_PATH)


print("\nInference completed!")


