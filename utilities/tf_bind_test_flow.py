from MCMC_sampler import MCMCSequenceSampler
from models.random_sampler import SequenceSampler
from tf_bind_8_oracle import tf_bind_8_oracle
import numpy as np
import pickle as pkl

with open('tests\\tf_bind_8_reward_proportionality_max_set.pkl', 'rb') as tp:
        reward_proportionality_max_set = pkl.load(tp)
        print('reward_proportionality_max_set')


with open('tests\\tf_bind_8_reward_proportionality_min_set.pkl', 'rb') as mp:
        reward_proportionality_min_set = pkl.load(mp)
        print('reward_proportionality_min_set')

with open('tests\\permutation_index.pkl', 'rb') as mp:
        index_to_permutation = pkl.load(mp)
        print('Permutation Index Dictionary Loaded')        

def sample_proportionality(samples):
    
    count_dict = {'min_set': 0, 'max_set': 0}

    for sample in samples:

        string_form = ''

        for element in sample:
            string_form += str(element)

        if string_form in reward_proportionality_min_set.keys():
            count_dict['min_set'] += 1

        if string_form in reward_proportionality_max_set:
            count_dict['max_set'] += 1

    return count_dict

if __name__ == "__main__":
    s = {'A':0, 'C':1, 'G':2, 'T':3}
    l = 8
    n = 100

    random_sampler = SequenceSampler(s, l, n)
    MCMC_sampler = MCMCSequenceSampler(s, l, n)

    MCMC_samples = MCMC_sampler.sample()
    random_samples = random_sampler.sample()

    print(MCMC_samples)
    print(random_samples)

    #Evaluate performance of samplers

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

    #Proportionality scores
    
    MCMC_proportionality = sample_proportionality(MCMC_samples)
    random_proportionality = sample_proportionality(random_samples)

    print(MCMC_proportionality.values())
    print(random_proportionality.values())

    









    