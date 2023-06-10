
import torch
import numpy as np

# Import scripts
from MCMC_sampler import MCMCSequenceSampler
from models.random_sampler import SequenceSampler
from models.train import train_flow_matching
from models.tfbind8_model import GFlowNet
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions import torch_helperfunctions as help
from utilities.loss_plot import loss_plot
from evaluation.evaluation import evaluate_modelsampling
from utilities.transformer import Transformer


# Hyperparameters
HIDDEN_SIZE = 2048
N_HIDDEN_LAYERS = 2

SAMPLE_SIZE = 100
BURNIN = 10
VERBOSE = True

# Data
X_TRAIN_PATH = "data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt"
MODEL_PATH = "models/saved_models/test_model_2048_400.tar"
REWARD_PATH = "models/saved_models/tfbind_reward_model_1.pt"

device = help.set_device()
X_train = torch.load(X_TRAIN_PATH)

# Sample random
random_sampler = SequenceSampler()
random_samples = random_sampler.sample_onehot(SAMPLE_SIZE)

# Sample MCMC
# MCMC_sampler = MCMCSequenceSampler()
# MCMC_samples = transformer.list_list_int_to_tensor_one_hot(MCMC_sampler.sample(SAMPLE_SIZE, BURNIN))

# Setup GFlow model and sample
model_dict = torch.load(MODEL_PATH, map_location=device)
model_state_dict = model_dict["model_state_dict"]
#n_hid, n_hidden_layers = model_dict["hidden_size"], model_dict["hidden_layers"] # TODO: Possible automatic initialization
#model = GFlowNet(hidden_size=hidden_size, hidden_layers=hidden_layers)
model = GFlowNet(n_hid=HIDDEN_SIZE, n_hidden_layers=N_HIDDEN_LAYERS)
model.to(device)
model.load_state_dict(model_state_dict)
Gflow_samples = model.sample(SAMPLE_SIZE)

# Setup reward function # TODO: Possibly replace reward function with oracle
reward_func = TFBindReward1HOT()
reward_func.to(device)
reward_func.load_state_dict(torch.load(REWARD_PATH, map_location=device))

# Predict
random_reward = reward_func(random_samples)
#MCMC_reward = reward_func(MCMC_samples)
gflow_reward = reward_func(Gflow_samples)

# Evaluate
random_evaluation = evaluate_modelsampling(X_train,random_samples,random_reward, print_stats = True)
#MCMC_evaluation = evaluate_modelsampling(X_train,MCMC_samples,MCMC_reward, print_stats = True)
gflow_evaluation = evaluate_modelsampling(X_train,Gflow_samples,gflow_reward, print_stats = True)

