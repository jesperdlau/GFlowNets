import numpy as np
from scipy.stats import t
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests 

def get_stats(data, alpha = 0.05, n_comparison = 1):
    performances = np.array([])
    diversities = np.array([])
    novelties = np.array([])
    N = len(data)
    alpha /= n_comparison

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


def compare_models(gflow, mcmc, random):
    """
	Compares the statistical significance of three sets of data using the t-test. 
	:param gflow: A 3xN numpy array representing one set of data.
	:param mcmc: A 3xN numpy array representing a second set of data.
	:param random: A 3xN numpy array representing a third set of data.
	:return: A 3x3 numpy array representing the p-values of the t-tests.

    returns =     np.array([[p_gfn_mcmc, p_gfn_random, p_mcmc_random],
                            [d_gfn_mcmc, d_gfn_random, d_mcmc_random],
                            [n_gfn_mcmc, n_gfn_random, n_mcmc_random]])
	"""

    result = np.zeros((3,3))

    for i in range(3):
        gfn_mcmc    = ttest_ind(gflow[i], mcmc[i]).pvalue
        gfn_random  = ttest_ind(gflow[i], random[i]).pvalue
        mcmc_random = ttest_ind(mcmc[i], random[i]).pvalue
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

if __name__ == "__main__":

    # data1 = np.load("evaluation/tfbind8_gflow_metrics_1.npy", allow_pickle=True)
    # data2 = np.load("evaluation/tfbind8_gflow_metrics_2.npy", allow_pickle=True)
    # data4 = np.load("evaluation/tfbind8_gflow_metrics_4.npy", allow_pickle=True)
    # data6 = np.load("evaluation/tfbind8_gflow_metrics_6.npy", allow_pickle=True)

    # data = [data1, data2, data4, data6]

    # print(get_stats(data, alpha = 0.05,n_comparison=1))

    a = np.random.rand(3,3)
    pvals, reject = correct_pvalues(a)

    print(pvals)
    print(reject)
    # print(correct_pvalues(a))
    