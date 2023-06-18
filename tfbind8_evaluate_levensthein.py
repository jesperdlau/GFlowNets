import numpy as np
from evaluation.statistics import get_stats_over_runs, get_stats, compare_models, correct_pvalues

# Load data for novelty and diversity
RANDOM_PATH = "evaluation/tfbind8_random_levenshtein_final10.npy"
MCMC_PATH = "evaluation/tfbind8_mcmc_levenshtein_final10.npy"
GFLOW_PATH = "evaluation/tfbind8_gflow_levenshtein_final10.npy"

random_metrics = np.load(RANDOM_PATH, allow_pickle=True)
mcmc_metrics = np.load(MCMC_PATH, allow_pickle=True)
gflow_metrics = np.load(GFLOW_PATH, allow_pickle=True)

rand_n = np.array([sample["Novelty_Levensthein_Last"] for sample in random_metrics])
mcmc_n = np.array([sample["Novelty_Levensthein_Last"] for sample in mcmc_metrics])
gflow_n = np.array([sample["Novelty_Levensthein_Last"] for sample in gflow_metrics])

rand_d = np.array([sample["Diversity_Levensthein_Last"] for sample in random_metrics])
mcmc_d = np.array([sample["Diversity_Levensthein_Last"] for sample in mcmc_metrics])
gflow_d = np.array([sample["Diversity_Levensthein_Last"] for sample in gflow_metrics])

# Load data for performance
GFLOW_BASE_PATH = "evaluation/tfbind8_gflow_metrics_"
runs = ["1", "2", "3", "4", "5", "6", "7", "8"]
paths = [GFLOW_BASE_PATH + run + ".npy" for run in runs]
models = [np.load(path, allow_pickle=True) for path in paths]
perfs = [model[-1]["Performance"] for model in models]
gflow_p = np.array(perfs)

RANDOM_10_PATH = "evaluation/tfbind8_random_metrics_final10.npy"
MCMC_10_PATH = "evaluation/tfbind8_mcmc_metrics_final10.npy"
random_10 = np.load(RANDOM_10_PATH, allow_pickle=True)
mcmc_10 = np.load(MCMC_10_PATH, allow_pickle=True)
rand_p = np.array([sample["Performance"] for sample in random_10])
mcmc_p = np.array([sample["Performance"] for sample in mcmc_10])

# Reformat all data to 3xN
gflow = np.vstack((gflow_p, gflow_d, gflow_n))
random = np.vstack((rand_p, rand_d, rand_n))
mcmc = np.vstack((mcmc_p, mcmc_d, mcmc_n))

# Get results
results = compare_models(gflow=gflow, mcmc=mcmc, random=random)
correct_results = correct_pvalues(results)

print(f"Results:\n{results}")
print(f"Corrected Results:\n{correct_results[0]}")
print(f"{correct_results[1]}")
print("Comparison complete.")