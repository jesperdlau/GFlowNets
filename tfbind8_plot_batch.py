import torch
import numpy as np
import scipy.stats as st
import pandas as pd

# Import scripts
from reward_functions import torch_helperfunctions as help
from utilities.plot_functions import loss_plot, eval_plot, combined_loss_eval_plot, combined_loss_eval_plot_flex, performance_plot, diversity_plot, novelty_plot, combined_plot
from evaluation.statistics import get_stats_over_runs, get_stats, compare_models, correct_pvalues
from utilities.plot_functions import plot_avg_over_runs

# Import Hyperparameters
from config.config import NAME_OF_RUN, PWD, PWD_WORK

#
NAME_OF_RUN = "final10"

# Load paths TODO: Import from shared config script?
RANDOM_METRICS_PATH = PWD + "evaluation/tfbind8_random_metrics_" + NAME_OF_RUN + ".npy"
# GFLOW_METRICS_PATH = PWD + "evaluation/tfbind8_gflow_metrics_" + RUN + ".npy"
MCMC_METRICS_PATH = PWD + "evaluation/tfbind8_mcmc_metrics_" + NAME_OF_RUN + ".npy"

# Random and mcmc
random_metrics = np.load(RANDOM_METRICS_PATH, allow_pickle=True)
mcmc_metrics = np.load(MCMC_METRICS_PATH, allow_pickle=True)
random_p_mean, random_d_mean, random_n_mean, random_p_CI, random_d_CI, random_n_CI = get_stats(random_metrics, episode=None)
mcmc_p_mean, mcmc_d_mean, mcmc_n_mean, mcmc_p_CI, mcmc_d_CI, mcmc_n_CI = get_stats(mcmc_metrics, episode=None)

# Gflow
runs = ["1", "2", "3", "4", "5", "6", "7", "8"]
gflow_run_metrics = get_stats_over_runs(runs, base_path="evaluation/tfbind8_gflow_metrics_")
gflow_p_list, gflow_d_list, gflow_n_list, gflow_p_CI_list, gflow_d_CI_list, gflow_n_CI_list = gflow_run_metrics
gflow_p_CI_y1, gflow_p_CI_y2 = np.array(gflow_p_CI_list)[:,0], np.array(gflow_p_CI_list)[:,1]
gflow_d_CI_y1, gflow_d_CI_y2 = np.array(gflow_d_CI_list)[:,0], np.array(gflow_d_CI_list)[:,1]
gflow_n_CI_y1, gflow_n_CI_y2 = np.array(gflow_n_CI_list)[:,0], np.array(gflow_n_CI_list)[:,1]

# Divided by 2 all across to get more correct Levensthein
plot_avg_over_runs(gflow_mean=gflow_p_list, gflow_ci=[gflow_p_CI_y1, gflow_p_CI_y2], 
                   mcmc_mean=mcmc_p_mean, mcmc_ci= mcmc_p_CI, 
                   random_mean=random_p_mean, random_ci=random_p_CI,
                   save_path="plots/tfbind8_gflow_performance_plot_lev.png", 
                   title="TFBind8 Performance over Training Steps", ylabel="Performance", running_mean=5)
plot_avg_over_runs(gflow_mean=[num/2 for num in gflow_d_list], gflow_ci=[[num/2 for num in gflow_d_CI_y1], [num/2 for num in gflow_d_CI_y2]], 
                   mcmc_mean=mcmc_d_mean/2, mcmc_ci=[num/2 for num in mcmc_d_CI], 
                   random_mean=random_d_mean/2, random_ci=[num/2 for num in random_d_CI],
                   save_path="plots/tfbind8_gflow_diversity_plot_lev.png", 
                   title="TFBind8 Diversity over Training Steps", ylabel="Diversity", running_mean=5)
plot_avg_over_runs(gflow_mean=[num/2 for num in gflow_n_list], gflow_ci=[[num/2 for num in gflow_n_CI_y1], [num/2 for num in gflow_n_CI_y2]], 
                   mcmc_mean=mcmc_n_mean/2, mcmc_ci=[num/2 for num in mcmc_n_CI], 
                   random_mean=random_n_mean/2, random_ci=[num/2 for num in random_n_CI],
                   save_path="plots/tfbind8_gflow_novelty_plot_lev.png", 
                   title="TFBind8 Novelty over Training Steps", ylabel="Novelty", running_mean=5)




