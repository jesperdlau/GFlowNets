
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
from evaluation.evaluation import evaluate_modelsampling
#from utilities.transformer import Transformer


# Hyperparameters
VERBOSE = True

# Data
X_TRAIN_PATH = "data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt"

# Load paths
RANDOM_SAMPLES_PATH = "inference/tfbind8_random_samples_400.pt"
RANDOM_REWARDS_PATH = "inference/tfbind8_random_rewards_400.pt"
GFLOW_SAMPLES_PATH = "inference/tfbind8_gflow_samples_400.pt"
GFLOW_REWARDS_PATH = "inference/tfbind8_gflow_rewards_400.pt"
#MCMC_SAMPLES_PATH = "inference/tfbind8_mcmc_samples_400.pt"
#MCMC_REWARDS_PATH = "inference/tfbind8_mcmc_rewards_400.pt"

# Save paths
RANDOM_METRICS_PATH = "inference/tfbind8_random_metrics_400.npy"
GFLOW_METRICS_PATH = "inference/tfbind8_gflow_metrics_400.npy"
#MCMC_METRICS_PATH = "inference/tfbind8_mcmc_metrics_400.npy"

# Load data
device = help.set_device()
X_train = torch.load(X_TRAIN_PATH)
random_samples = torch.load(RANDOM_SAMPLES_PATH)
random_rewards = torch.load(RANDOM_REWARDS_PATH)
gflow_samples = torch.load(GFLOW_SAMPLES_PATH)
gflow_rewards = torch.load(GFLOW_REWARDS_PATH)
# mcmc_samples = torch.load(MCMC_SAMPLES_PATH)
# mcmc_rewards = torch.load(MCMC_REWARDS_PATH)

# Get top 20% of random samples, then evaluate
mask = torch.argsort(random_rewards, dim=0, descending=True)
random_samples_sorted = random_samples[mask].squeeze()
random_rewards_sorted = random_rewards[mask].squeeze()
random_samples_top20 = random_samples_sorted[:int(len(random_samples_sorted) * 0.2)]
random_rewards_top20 = random_rewards_sorted[:int(len(random_rewards_sorted) * 0.2)]

random_perf, random_div, random_novel = evaluate_modelsampling(X_train,random_samples_top20,random_rewards_top20, print_stats = False)
random_metrics = np.array([{"Performance": random_perf.detach().numpy(), "Diversity": random_div.detach().numpy(), "Novelty": random_novel.detach().numpy()}])

# Get top 20% of MCMC samples, then evaluate
# mask = torch.argsort(mcmc_rewards, dim=0, descending=True)
# mcmc_samples_sorted = mcmc_samples[mask].squeeze()
# mcmc_rewards_sorted = random_rewards[mask].squeeze()
# mcmc_samples_top20 = mcmc_samples_sorted[:int(len(mcmc_samples_sorted) * 0.2)]
# mcmc_rewards_top20 = mcmc_rewards_sorted[:int(len(mcmc_rewards_sorted) * 0.2)]

# mcmc_perf, mcmc_div, mcmc_novel = evaluate_modelsampling(X_train,mcmc_samples_top20,mcmc_rewards_top20, print_stats = False)
# mcmc_metrics = np.array([{"Performance": mcmc_perf.detach().numpy(), "Diversity": mcmc_div.detach().numpy(), "Novelty": mcmc_novel.detach().numpy()}])


# Get top 20% of each gflow batch, then evaluate
gflow_metrics = []
for i in range(len(gflow_samples)):
    mask = torch.argsort(gflow_rewards[i], dim=0, descending=True)
    gflow_samples_sorted = gflow_samples[i][mask].squeeze()
    gflow_rewards_sorted = gflow_rewards[i][mask].squeeze()
    gflow_samples_top20 = gflow_samples_sorted[:int(len(gflow_samples_sorted) * 0.2)]
    gflow_rewards_top20 = gflow_rewards_sorted[:int(len(gflow_rewards_sorted) * 0.2)]
    gflow_perf, gflow_div, gflow_novel = evaluate_modelsampling(X_train,gflow_samples_top20,gflow_rewards_top20, print_stats = False)
    gflow_metrics.append({"Performance": gflow_perf.detach().numpy(), "Diversity": gflow_div.detach().numpy(), "Novelty": gflow_novel.detach().numpy()})
gflow_metrics = np.array(gflow_metrics)



# Save metrics
np.save(RANDOM_METRICS_PATH, random_metrics)
np.save(GFLOW_METRICS_PATH, gflow_metrics)
#np.save(MCMC_METRICS_PATH, mcmc_metrics)

print("Evaluation complete. Saved metrics.")