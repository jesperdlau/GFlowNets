import torch
import numpy as np
import scipy.stats as st

# Import scripts
from reward_functions import torch_helperfunctions as help
from utilities.plot_functions import loss_plot, eval_plot, combined_loss_eval_plot, combined_loss_eval_plot_flex, performance_plot, diversity_plot, novelty_plot, combined_plot

# Import Hyperparameters
from config.config import NAME_OF_RUN, PWD, PWD_WORK

# Model path (For loading losses)
MODEL_PATH = PWD + "models/saved_models/tfbind8_gflow_model_" + NAME_OF_RUN + ".tar"

# Load paths TODO: Import from shared config script?
RANDOM_METRICS_PATH = PWD + "inference/tfbind8_random_metrics_" + NAME_OF_RUN + ".npy"
GFLOW_METRICS_PATH = PWD + "inference/tfbind8_gflow_metrics_" + NAME_OF_RUN + ".npy"
MCMC_METRICS_PATH = PWD + "inference/tfbind8_mcmc_metrics_" + NAME_OF_RUN + ".npy"

# Save paths for plots
# PLOT_LOSS_PATH = "plots/tfbind8_loss_plot_" + NAME_OF_RUN + ".png"
# PLOT_EVALUATION_PATH = "plots/tfbind8_eval_plot_" + NAME_OF_RUN + ".png"
# PLOT_COMBINED_PATH = "plots/tfbind8_combined_plot_" + NAME_OF_RUN + ".png"

# Save paths
PLOT_PERFORMANCE_PATH = PWD + "plots/tfbind8_performance_plot_" + NAME_OF_RUN + ".png"
PLOT_DIVERSITY_PATH = PWD + "plots/tfbind8_diversity_plot_" + NAME_OF_RUN + ".png"
PLOT_NOVELTY_PATH = PWD + "plots/tfbind8_novelty_plot_" + NAME_OF_RUN + ".png"

# Load losses
device = help.set_device()
model_dict = torch.load(MODEL_PATH, map_location=device)
losses = model_dict["minibatch_loss"]

# Load metrics
random_metrics = np.load(RANDOM_METRICS_PATH, allow_pickle=True)
gflow_metrics = np.load(GFLOW_METRICS_PATH, allow_pickle=True)
mcmc_metrics = np.load(MCMC_METRICS_PATH, allow_pickle=True)

# Isolate individual metrics
random_perf = [random_metrics[i]["Performance"] for i in range(len(random_metrics))]
random_div = [random_metrics[i]["Diversity"] for i in range(len(random_metrics))]
random_novel = [random_metrics[i]["Novelty"] for i in range(len(random_metrics))]

mcmc_perf = [mcmc_metrics[i]["Performance"] for i in range(len(mcmc_metrics))]
mcmc_div = [mcmc_metrics[i]["Diversity"] for i in range(len(mcmc_metrics))]
mcmc_novel = [mcmc_metrics[i]["Novelty"] for i in range(len(mcmc_metrics))]

gflow_perf = [gflow_metrics[i]["Performance"] for i in range(len(gflow_metrics))]
gflow_div = [gflow_metrics[i]["Diversity"] for i in range(len(gflow_metrics))]
gflow_novel = [gflow_metrics[i]["Novelty"] for i in range(len(gflow_metrics))]

# Means
random_perf_mean, random_div_mean, random_novel_mean = np.mean(random_perf), np.mean(random_div), np.mean(random_novel)
mcmc_perf_mean, mcmc_div_mean, mcmc_novel_mean = np.mean(mcmc_perf), np.mean(mcmc_div), np.mean(mcmc_novel)

# Get 95% confidence intervals for random and MCMC metrics
random_perf_ci = st.t.interval(confidence=0.95, df=len(random_perf)-1, loc=np.mean(random_perf), scale=st.sem(random_perf))
random_div_ci = st.t.interval(confidence=0.95, df=len(random_div)-1, loc=np.mean(random_div), scale=st.sem(random_div))
random_novel_ci = st.t.interval(confidence=0.95, df=len(random_novel)-1, loc=np.mean(random_novel), scale=st.sem(random_novel))

mcmc_perf_ci = st.t.interval(confidence=0.95, df=len(mcmc_perf)-1, loc=np.mean(mcmc_perf), scale=st.sem(mcmc_perf))
mcmc_div_ci = st.t.interval(confidence=0.95, df=len(mcmc_div)-1, loc=np.mean(mcmc_div), scale=st.sem(mcmc_div))
mcmc_novel_ci = st.t.interval(confidence=0.95, df=len(mcmc_novel)-1, loc=np.mean(mcmc_novel), scale=st.sem(mcmc_novel))

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
# performance_plot(losses=losses, random_perf=random_perf, gflow_perf=gflow_perf, save_path=PLOT_PERFORMANCE_PATH)
# diversity_plot(losses=losses, random_diversity=random_div, gflow_diversity=gflow_div, save_path=PLOT_DIVERSITY_PATH)
# novelty_plot(losses=losses, random_novel=random_novel, gflow_novel=gflow_novel, save_path=PLOT_NOVELTY_PATH)

# Combined plots
combined_plot(losses=losses, 
              random_mean=random_perf_mean, 
              random_ci=random_perf_ci, 
              mcmc_mean=mcmc_perf_mean, 
              mcmc_ci=mcmc_perf_ci, 
              gflow_data = gflow_perf, 
              save_path=PLOT_PERFORMANCE_PATH, 
              plot_type="Performance")

combined_plot(losses=losses, 
              random_mean=random_perf_mean, 
              random_ci=random_div_ci, 
              mcmc_mean=mcmc_div_mean, 
              mcmc_ci=mcmc_div_ci, 
              gflow_data = gflow_div, 
              save_path=PLOT_DIVERSITY_PATH, 
              plot_type="Diversity")

combined_plot(losses=losses, 
              random_mean=random_novel_mean, 
              random_ci=random_novel_ci, 
              mcmc_mean=mcmc_novel_mean, 
              mcmc_ci=mcmc_novel_ci, 
              gflow_data = gflow_novel, 
              save_path=PLOT_NOVELTY_PATH, 
              plot_type="Novelty")



