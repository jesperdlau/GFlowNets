
import torch
import numpy as np

# Import scripts
#from MCMC_sampler import MCMCSequenceSampler
#from models.random_sampler import SequenceSampler
#from models.train import train_flow_matching
#from models.tfbind8_model import GFlowNet
#from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
from reward_functions import torch_helperfunctions as help
from utilities.plot_functions import loss_plot, eval_plot, combined_loss_eval_plot, combined_loss_eval_plot_flex, performance_plot, diversity_plot, novelty_plot
#from evaluation.evaluation import evaluate_modelsampling
#from utilities.transformer import Transformer


device = help.set_device()

# Data
MODEL_PATH = "models/saved_models/test_model_2048_400_3.tar"

RANDOM_METRICS_PATH = "inference/tfbind8_random_metrics_400.npy"
GFLOW_METRICS_PATH = "inference/tfbind8_gflow_metrics_400.npy"
#MCMC_METRICS_PATH = "inference/tfbind8_mcmc_metrics_400.npy"

# Save paths for plots
PLOT_LOSS_PATH = "plots/tfbind8_loss_plot_400.png"
PLOT_EVALUATION_PATH = "plots/tfbind8_eval_plot_400.png"
PLOT_COMBINED_PATH = "plots/tfbind8_combined_plot_400.png"

PLOT_PERFORMANCE_PATH = "plots/tfbind8_performance_plot_400.png"
PLOT_DIVERSITY_PATH = "plots/tfbind8_diversity_plot_400.png"
PLOT_NOVELTY_PATH = "plots/tfbind8_novelty_plot_400.png"

# Load
model_dict = torch.load(MODEL_PATH, map_location=device)
losses = model_dict["losses"]

random_metrics = np.load(RANDOM_METRICS_PATH, allow_pickle=True)
gflow_metrics = np.load(GFLOW_METRICS_PATH, allow_pickle=True)
#mcmc_metrics = np.load(MCMC_METRICS_PATH, allow_pickle=True)

# Get individual metrics
random_perf, random_div, random_novel = random_metrics[0]["Performance"], random_metrics[0]["Diversity"], random_metrics[0]["Novelty"]
#mcmc_perf, mcmc_div, mcmc_novel = mcmc_metrics[0]["Performance"], mcmc_metrics[0]["Diversity"], mcmc_metrics[0]["Novelty"]

gflow_perf = [gflow_metrics[i]["Performance"] for i in range(len(gflow_metrics))]
gflow_div = [gflow_metrics[i]["Diversity"] for i in range(len(gflow_metrics))]
gflow_novel = [gflow_metrics[i]["Novelty"] for i in range(len(gflow_metrics))]


### Create different plots

# Plot log-loss only
#loss_plot(losses, save_path=PLOT_LOSS_PATH)

# Plot evaluation metrics only
#eval_plot(gflow_perf, gflow_div, gflow_novel, save_path=PLOT_EVALUATION_PATH)

# Plot combined
#combined_loss_eval_plot(losses, perfs, divs, novels, save_path=PLOT_COMBINED_PATH)

# Plot combined, with flexible amount of metrics. Just insert None to metrics if needed. 
#combined_loss_eval_plot_flex(losses, perfs=perfs, divs=None, novels=novels, save_path=PLOT_COMBINED_PATH)

# Individual metric plots over batch with overlay
performance_plot(losses=losses, random_perf=random_perf, gflow_perf=gflow_perf, save_path=PLOT_PERFORMANCE_PATH)
diversity_plot(losses=losses, random_diversity=random_div, gflow_diversity=gflow_div, save_path=PLOT_DIVERSITY_PATH)
novelty_plot(losses=losses, random_novel=random_novel, gflow_novel=gflow_novel, save_path=PLOT_NOVELTY_PATH)

