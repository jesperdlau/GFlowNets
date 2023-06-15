import numpy as np
from scipy.stats import t


def get_stats(data, alpha = 0.5, n_comparison = 3):
    performances = np.array([])
    diversities = np.array([])
    novelties = np.array([])
    N = len(data)
    alpha /= 3

    for model in data:
        performances = np.append(performances, model[-1]["Performance"])
        diversities  = np.append(diversities, model[-1]["Diversity"])
        novelties    = np.append(novelties, model[-1]["Novelty"])

    p_mean = performances.mean()
    d_mean = diversities.mean()
    n_mean = novelties.mean()

    p_std = performances.std()
    d_std = diversities.std()
    n_std = novelties.std()

    p_std_error = p_std/np.sqrt(N)
    d_std_error = d_std/np.sqrt(N)
    n_std_error = n_std/np.sqrt(N)

    t_value = t.ppf(1 - alpha/2, df = N - 1)

    p_CI = [p_mean - t_value * p_std_error, p_mean + t_value * p_std_error]
    d_CI = [d_mean - t_value * d_std_error, d_mean + t_value * d_std_error]
    n_CI = [n_mean - t_value * n_std_error, n_mean + t_value * n_std_error]
    
    return p_mean, d_mean, n_mean, p_CI, d_CI, n_CI


if __name__ == "__main__":

    data1 = np.load("evaluation/tfbind8_gflow_metrics_1.npy", allow_pickle=True)
    data2 = np.load("evaluation/tfbind8_gflow_metrics_2.npy", allow_pickle=True)
    data4 = np.load("evaluation/tfbind8_gflow_metrics_4.npy", allow_pickle=True)
    data6 = np.load("evaluation/tfbind8_gflow_metrics_6.npy", allow_pickle=True)

    data = [data1, data2, data4, data6]

    print(get_stats(data, alpha = 0.05,n_comparison=3))


    