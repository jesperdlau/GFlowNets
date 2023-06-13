
import torch
import numpy as np
import pandas as pd

# Import scripts
#from MCMC_sampler import MCMCSequenceSampler
#from models.random_sampler import SequenceSampler
#from models.train import train_flow_matching
#from models.tfbind8_model import GFlowNet
#from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions import torch_helperfunctions as help
#from utilities.plot_functions import loss_plot
from evaluation.evaluation import evaluate_modelsampling, get_top20percent, evaluate_batches
#from utilities.transformer import Transformer

# Import Hyperparameters
import config
NAME_OF_RUN = config.NAME_OF_RUN
PWD = config.PWD
PWD_WORK = config.PWD_WORK

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

# Get top 20% of random samples, then evaluate
# mask = torch.argsort(random_rewards, dim=0, descending=True)
# random_samples_sorted = random_samples[mask].squeeze()
# random_rewards_sorted = random_rewards[mask].squeeze()
# random_samples_top20 = random_samples_sorted[:int(len(random_samples_sorted) * 0.2)]
# random_rewards_top20 = random_rewards_sorted[:int(len(random_rewards_sorted) * 0.2)]

# random_perf, random_div, random_novel = evaluate_modelsampling(X_train,random_samples_top20,random_rewards_top20, print_stats = False)
# random_metrics = np.array([{"Performance": random_perf.detach().numpy().item(), "Diversity": random_div.detach().numpy().item(), "Novelty": random_novel.detach().numpy().item()}])

# Get top 20% of MCMC samples, then evaluate
# mask = torch.argsort(mcmc_rewards, dim=0, descending=True)
# mcmc_samples_sorted = mcmc_samples[mask].squeeze()
# mcmc_rewards_sorted = random_rewards[mask].squeeze()
# mcmc_samples_top20 = mcmc_samples_sorted[:int(len(mcmc_samples_sorted) * 0.2)]
# mcmc_rewards_top20 = mcmc_rewards_sorted[:int(len(mcmc_rewards_sorted) * 0.2)]

# mcmc_perf, mcmc_div, mcmc_novel = evaluate_modelsampling(X_train,mcmc_samples_top20,mcmc_rewards_top20, print_stats = False)
# mcmc_metrics = np.array([{"Performance": mcmc_perf.detach().numpy(), "Diversity": mcmc_div.detach().numpy(), "Novelty": mcmc_novel.detach().numpy()}])


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