
import torch
import numpy as np

# Import scripts
from MCMC_sampler import MCMCSequenceSampler
from models.random_sampler import SequenceSampler
from models.train import train_flow_matching
from models.tfbind8_model import GFlowNet
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions import torch_helperfunctions as help
from utilities.plot_functions import loss_plot, eval_plot, combined_loss_eval_plot, combined_loss_eval_plot_flex
from evaluation.evaluation import evaluate_modelsampling
from utilities.transformer import Transformer


# Hyperparameters
#HIDDEN_SIZE = 2048
#N_HIDDEN_LAYERS = 2

SAMPLE_SIZE = 100
BURNIN = 10
VERBOSE = True

# Data
X_TRAIN_PATH = "data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt"
MODEL_PATH = "models/saved_models/test_model_2048.tar"
REWARD_PATH = "models/saved_models/tfbind_reward_model_1.pt"

# Save paths for plots
PLOT_LOSS_PATH = "plots/tfbind8_loss_plot_test.png"
PLOT_EVALUATION_PATH = "plots/tfbind8_eval_plot_test.png"
PLOT_COMBINED_PATH = "plots/tfbind8_combined_plot_test.png"

# Load
device = help.set_device()
X_train = torch.load(X_TRAIN_PATH)
model_dict = torch.load(MODEL_PATH, map_location=device)
losses = model_dict["losses"]
models = model_dict["models"]

# Sample random
#random_sampler = SequenceSampler()
#random_samples = random_sampler.sample_onehot(SAMPLE_SIZE)

# Sample MCMC
# MCMC_sampler = MCMCSequenceSampler()
# MCMC_samples = transformer.list_list_int_to_tensor_one_hot(MCMC_sampler.sample(SAMPLE_SIZE, BURNIN))

# Setup reward function # TODO: Possibly replace reward function with oracle
reward_func = TFBindReward1HOT()
reward_func.to(device)
reward_func.load_state_dict(torch.load(REWARD_PATH, map_location=device))

# Setup GFlow model
n_hid, n_hidden_layers = model_dict["n_hid"], model_dict["n_hidden_layers"] 
model = GFlowNet(n_hid=n_hid, n_hidden_layers=n_hidden_layers)
model.to(device)

# If only the latest, greatest model is to be loaded
#model_state_dict = model_dict["model_state_dict"]
#model.load_state_dict(model_state_dict)
#Gflow_samples = model.sample(SAMPLE_SIZE)

# Get evaluation metrics for each model 
perfs, divs, novels = [], [], []
for model_state in models:
    model.load_state_dict(model_state)
    Gflow_samples = model.sample(SAMPLE_SIZE)
    gflow_reward = reward_func(Gflow_samples)
    perf, div, novel = evaluate_modelsampling(X_train,Gflow_samples,gflow_reward, print_stats = False)
    perfs.append(perf.detach().numpy())
    divs.append(div.detach().numpy())
    novels.append(novel.detach().numpy())


# Plot log-loss only
loss_plot(losses, save_path=PLOT_LOSS_PATH)

# Plot evaluation metrics only
eval_plot(perfs, divs, novels, save_path=PLOT_EVALUATION_PATH)

# Plot combined
#combined_loss_eval_plot(losses, perfs, divs, novels, save_path=PLOT_COMBINED_PATH)

# Plot combined, with flexible amount of metrics. Just insert None to metrics if needed. 
combined_loss_eval_plot_flex(losses, perfs=perfs, divs=None, novels=novels, save_path=PLOT_COMBINED_PATH)

print()