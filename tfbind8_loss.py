import torch
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# Import scripts
from reward_functions import torch_helperfunctions as help
from utilities.plot_functions import loss_plot, eval_plot, combined_loss_eval_plot, combined_loss_eval_plot_flex, performance_plot, diversity_plot, novelty_plot, combined_plot

# Paths
path = "/home/jesper/Documents/Fagprojekt/GFlowNets/evaluation/loss_arr.npy"
save_path = "/home/jesper/Documents/Fagprojekt/GFlowNets/plots/loss.png"

# Load
arr = np.load(path)
arr_mean = arr.mean(axis=0)

# Plot
loss_plot(arr_mean, save_path)

