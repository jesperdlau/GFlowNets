import numpy as np
from scipy.stats import t
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests 


def get_stats(data, episode=-1, alpha = 0.05, n_comparison = 3):
    performances = np.array([])
    diversities = np.array([])
    novelties = np.array([])
    N = len(data)
    alpha /= n_comparison

    for model in data:
        performances = np.append(performances, model[episode]["Performance"])
        diversities  = np.append(diversities, model[episode]["Diversity"])
        novelties    = np.append(novelties, model[episode]["Novelty"])

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


def compare_models(gflow, mcmc, random, parametric = False):
    """
	Compares the statistical significance of three sets of data using the t-test. 
	:param gflow: A 3xN numpy array representing one set of data.
	:param mcmc: A 3xN numpy array representing a second set of data.
	:param random: A 3xN numpy array representing a third set of data.
    :parametric: if True, perform a parametric t-test. If false, perform mannwhitneyu test.
	:return: A 3x3 numpy array representing the p-values of the t-tests.

    returns =     np.array([[p_gfn_mcmc, p_gfn_random, p_mcmc_random],
                            [d_gfn_mcmc, d_gfn_random, d_mcmc_random],
                            [n_gfn_mcmc, n_gfn_random, n_mcmc_random]])
	"""

    result = np.zeros((3,3))

    for i in range(3):
        if parametric:
            gfn_mcmc    = ttest_ind(gflow[i], mcmc[i],   equal_var = False).pvalue
            gfn_random  = ttest_ind(gflow[i], random[i], equal_var = False).pvalue
            mcmc_random = ttest_ind(mcmc[i], random[i],  equal_var = False).pvalue

        else:
            gfn_mcmc    = mannwhitneyu(gflow[i], mcmc[i], method = "exact").pvalue
            gfn_random  = mannwhitneyu(gflow[i], random[i], method = "exact").pvalue
            mcmc_random = mannwhitneyu(mcmc[i], random[i], method = "exact").pvalue
    
        result[i,0] = gfn_mcmc
        result[i,1] = gfn_random
        result[i,2] = mcmc_random

    return result


def correct_pvalues(result):
    """
    Perform false discovery rate correction on p-values using the Benjamini-Hochberg method.

    Args:
    - results (numpy.ndarray): A 2D array of p-values to be corrected.

    Returns:
    - pvals (numpy.ndarray): A 2D array of corrected p-values.
    - reject (numpy.ndarray): A boolean 2D array indicating which hypothesis tests are rejected.
    """

    shape = result.shape
    result = result.flatten()
    mpt = multipletests(result, alpha=0.05, method='fdr_bh')
    reject = mpt[0]
    pvals = mpt[1]
    reject = reject.reshape(*shape)
    pvals = pvals.reshape(*shape)

    return pvals, reject

# Runs e.g.: list of strings: ["1", "2", "4", "6"]
# base_path e.g.: string: "tfbind8_gflow_metrics_"
def get_stats_over_runs(runs, base_path):
    data_list = []
    for run in runs:
        data = np.load(base_path + run + ".npy", allow_pickle=True)
        data_list.append(data)

    p_mean_list, d_mean_list, n_mean_list = [], [], []
    p_CI_list, d_CI_list, n_CI_list = [], [], []

    for episode in range(100):
        p_mean, d_mean, n_mean, p_CI, d_CI, n_CI = get_stats(data_list, episode=episode, alpha=0.05, n_comparison=1)
        p_mean_list.append(p_mean)
        d_mean_list.append(d_mean)
        n_mean_list.append(n_mean)
        p_CI_list.append(p_CI)
        d_CI_list.append(d_CI)
        n_CI_list.append(n_CI)

    return p_mean_list, d_mean_list, n_mean_list, p_CI_list, d_CI_list, n_CI_list



# from: https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)



if __name__ == "__main__":

    data1 = np.load("evaluation/tfbind8_gflow_metrics_1.npy", allow_pickle=True)
    data2 = np.load("evaluation/tfbind8_gflow_metrics_2.npy", allow_pickle=True)
    data4 = np.load("evaluation/tfbind8_gflow_metrics_4.npy", allow_pickle=True)
    data6 = np.load("evaluation/tfbind8_gflow_metrics_6.npy", allow_pickle=True)

    data = [data1, data2, data4, data6]

    print(get_stats(data, alpha = 0.05,n_comparison=1))

    # a = np.random.rand(3,3)
    # print(a)
    # pvals, reject = correct_pvalues(a)

    # print(pvals)
    # print(reject)
    # print(bmatrix(pvals))
    