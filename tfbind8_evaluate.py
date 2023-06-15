
import torch
import numpy as np
import pandas as pd

# Import scripts
from reward_functions import torch_helperfunctions as help
from evaluation.evaluation import evaluate_modelsampling, get_top20percent, evaluate_batches

# Import Hyperparameters
from config.config import NAME_OF_RUN, PWD, PWD_WORK

# Data
X_TRAIN_PATH = PWD_WORK + "data/tfbind8/tfbind8_X_train.pt"

# Load paths 
RANDOM_SAMPLES_PATH = PWD_WORK + "inference/tfbind8_random_samples_" + NAME_OF_RUN + ".pt"
RANDOM_REWARDS_PATH = PWD_WORK + "inference/tfbind8_random_rewards_" + NAME_OF_RUN + ".pt"
GFLOW_SAMPLES_PATH = PWD_WORK + "inference/tfbind8_gflow_samples_" + NAME_OF_RUN + ".pt"
GFLOW_REWARDS_PATH = PWD_WORK + "inference/tfbind8_gflow_rewards_" + NAME_OF_RUN + ".pt"
MCMC_SAMPLES_PATH = PWD_WORK + "inference/tfbind8_mcmc_samples_" + NAME_OF_RUN + ".pt"
MCMC_REWARDS_PATH = PWD_WORK + "inference/tfbind8_mcmc_rewards_" + NAME_OF_RUN + ".pt"

# Save paths 
RANDOM_METRICS_PATH = PWD + "inference/tfbind8_random_metrics_" + NAME_OF_RUN + ".npy"
GFLOW_METRICS_PATH = PWD + "inference/tfbind8_gflow_metrics_" + NAME_OF_RUN + ".npy"
MCMC_METRICS_PATH = PWD + "inference/tfbind8_mcmc_metrics_" + NAME_OF_RUN + ".npy"

# Setup
device = help.set_device()
X_train = torch.load(X_TRAIN_PATH)

# Load data
random_samples = torch.load(RANDOM_SAMPLES_PATH)
random_rewards = torch.load(RANDOM_REWARDS_PATH)
gflow_samples = torch.load(GFLOW_SAMPLES_PATH)
gflow_rewards = torch.load(GFLOW_REWARDS_PATH)
mcmc_samples = torch.load(MCMC_SAMPLES_PATH)
mcmc_rewards = torch.load(MCMC_REWARDS_PATH)

# Get top 20% of each gflow batch, then evaluate
random_metrics = evaluate_batches(X_sampled = random_samples, y_sampled = random_rewards, X_train = X_train, print_stats=False)
print("Random metrics completed")
mcmc_metrics = evaluate_batches(X_sampled = mcmc_samples, y_sampled = mcmc_rewards, X_train = X_train, print_stats=False)
print("MCMC metrics completed")
gflow_metrics = evaluate_batches(X_sampled = gflow_samples, y_sampled = gflow_rewards, X_train = X_train, print_stats=False)
print("GFlow metrics completed")


# Save metrics
np.save(RANDOM_METRICS_PATH, random_metrics, allow_pickle=True)
np.save(GFLOW_METRICS_PATH, gflow_metrics, allow_pickle=True)
np.save(MCMC_METRICS_PATH, mcmc_metrics, allow_pickle=True)

print("Evaluation complete. Saved metrics.")