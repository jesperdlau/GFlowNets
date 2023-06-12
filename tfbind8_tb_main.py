
import torch
import numpy as np

# Import scripts
from MCMC_sampler import MCMCSequenceSampler
from models.random_sampler import SequenceSampler
from models.train import train_flow_matching
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
REWARD_PATH = "models/saved_models/tfbind_reward_earlystopping.pth"
PLOT_PATH = "models/test_loss_plot_3.png" # TODO: Organize model into model folder, reward into reward folder and plot into plot folder???
SAMPLE_SIZE = 100
BURNIN = 10
DATA_FOLDER = "data/"

# Set device
device = help.set_device()

# Load model and optimizer
model = GFlowNet(HIDDEN_SIZE)
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

# Load reward function
reward_func = TFBindReward1HOT()

# Train model and save checkpoint to PATH
train_flow_matching(model, optimizer, reward_func, SEQ_LEN, NUM_EPISODES, UPDATE_FREQ, MODEL_PATH, REWARD_PATH, device, hot_start=False, verbose=True)

# Plot of loss saved to PLOT_PATH
loss_plot(MODEL_PATH, save_path=PLOT_PATH)

# Sample
MCMC_sampler = MCMCSequenceSampler()
random_sampler = SequenceSampler()

random_samples = Transformer.list_list_int_to_tensor_one_hot(random_sampler.sample(SAMPLE_SIZE))
#MCMC_samples = transformer.list_list_int_to_tensor_one_hot(MCMC_sampler.sample(SAMPLE_SIZE, BURNIN))
Gflow_samples = Transformer.list_list_int_to_tensor_one_hot(model.sample(SAMPLE_SIZE))

#predict

random_reward = reward_func(random_samples)
#MCMC_reward = reward_func(MCMC_samples)
gflow_reward = reward_func(Gflow_samples)

#Evaluate

X_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt")

random_evaluation = evaluate_modelsampling(X_train,random_samples,random_reward, print_stats = True)
#MCMC_evaluation = evaluate_modelsampling(X_train,MCMC_samples,MCMC_reward, print_stats = True)
gflow_evaluation = evaluate_modelsampling(X_train,Gflow_samples,gflow_reward, print_stats = True)




