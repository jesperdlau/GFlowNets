from models.MCMC_sampler import MCMCSequenceSampler
from models.random_sampler import SequenceSampler
from tf_bind_8_oracle import tf_bind_8_oracle
import numpy as np

if __name__ == "__main__":
    s = {'A':0, 'C':1, 'G':2, 'T':3}
    l = 8
    n = 128

    random_sampler = SequenceSampler(s, l, n)
    MCMC_sampler = MCMCSequenceSampler(s, l, n)

    MCMC_samples = MCMC_sampler.sample()
    random_samples = random_sampler.sample()

    print(MCMC_samples)
    print(random_samples)

    #Evaluate samples

    oracle = tf_bind_8_oracle()

    pred_MCMC = oracle.predict(MCMC_samples)
    pred_random = oracle.predict(random_samples)

    print(pred_MCMC)
    print(pred_random)

    #Average scores
    average_MCMC = sum(pred_MCMC)/len(pred_MCMC)
    average_random = sum(pred_random)/len(pred_random)

    print(average_MCMC)
    print(average_random)

    









    