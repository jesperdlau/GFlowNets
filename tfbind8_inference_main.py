
import torch
import numpy as np
import pandas as pd

# Import scripts
from MCMC_sampler import MCMCSequenceSampler
from models.random_sampler import SequenceSampler
#from models.train import train_flow_matching
from models.tfbind8_model import GFlowNet
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions import torch_helperfunctions as help
#from utilities.plot_functions import loss_plot, eval_plot, combined_loss_eval_plot, combined_loss_eval_plot_flex
#from evaluation.evaluation import evaluate_modelsampling
from utilities.transformer import Transformer


# Hyperparameters
SAMPLE_SIZE = 100
BURNIN = 10
VERBOSE = True

# Data
X_TRAIN_PATH = "data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt"
MODEL_PATH = "models/saved_models/test_model_2048_400_3.tar"
REWARD_PATH = "models/saved_models/tfbind_reward_model_1.pt"

# Save paths
RANDOM_SAMPLES_PATH = "inference/tfbind8_random_samples_400.pt"
RANDOM_REWARDS_PATH = "inference/tfbind8_random_rewards_400.pt"
GFLOW_SAMPLES_PATH = "inference/tfbind8_gflow_samples_400.pt"
GFLOW_REWARDS_PATH = "inference/tfbind8_gflow_rewards_400.pt"
#MCMC_SAMPLES_PATH = "inference/tfbind8_mcmc_samples_400.pt"
#MCMC_REWARDS_PATH = "inference/tfbind8_mcmc_rewards_400.pt"

# Load
device = help.set_device()
X_train = torch.load(X_TRAIN_PATH)
model_dict = torch.load(MODEL_PATH, map_location=device)
models = model_dict["models"]

# Setup reward function # TODO: Possibly replace reward function with oracle
reward_func = TFBindReward1HOT()
reward_func.to(device)
reward_func.load_state_dict(torch.load(REWARD_PATH, map_location=device))
reward_func.eval()

# Sample MCMC
# MCMC_sampler = MCMCSequenceSampler()
# MCMC_samples = transformer.list_list_int_to_tensor_one_hot(MCMC_sampler.sample(SAMPLE_SIZE, BURNIN))

# Sample random
random_sampler = SequenceSampler()
random_samples = random_sampler.sample_onehot(SAMPLE_SIZE)
random_rewards = reward_func(random_samples) # TODO: Device?

# Setup GFlow model
n_hid, n_hidden_layers = model_dict["n_hid"], model_dict["n_hidden_layers"] 
model = GFlowNet(n_hid=n_hid, n_hidden_layers=n_hidden_layers)
model.to(device)

# Loop over gflow models, sample and concat to dataframe
df_gflow = pd.DataFrame()
gflow_samples_list, gflow_rewards_list = [], []
for i, model_state in enumerate(models):
    print(f"#{i+1} / {len(models)}")
    model.load_state_dict(model_state)
    model.eval()

    gflow_batch_samples = model.sample(SAMPLE_SIZE) # Note: This is the most time consuming step
    gflow_batch_reward = reward_func(gflow_batch_samples)

    gflow_samples_list.append(gflow_batch_samples)
    gflow_rewards_list.append(gflow_batch_reward)

# Convert to torch
gflow_samples_pt = torch.stack(gflow_samples_list, dim=0)
gflow_rewards_pt = torch.stack(gflow_rewards_list, dim=0)

# Save 
torch.save(random_samples, RANDOM_SAMPLES_PATH)
torch.save(random_rewards, RANDOM_REWARDS_PATH)
torch.save(gflow_samples_pt, GFLOW_SAMPLES_PATH)
torch.save(gflow_rewards_pt, GFLOW_REWARDS_PATH)
#torch.save(mcmc_samples_pt, MCMC_SAMPLES_PATH)
#torch.save(mcmc_rewards_pt, MCMC_REWARDS_PATH)


print("\nInference completed!")


