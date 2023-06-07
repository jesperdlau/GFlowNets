import random
import numpy as np
import pickle as pkl
from tf_bind_8_oracle import tf_bind_8_oracle
from scipy.stats import norm
import matplotlib.pyplot as plt

def string_to_list_int(string):
        sequence = []
        for element in string:
            sequence.append(int(element))

        return sequence
    
def list_int_to_string(list_int):
    string = ''
    for element in list_int:
        string += str(element)
    return string

def bounds(integer):
    lower = integer - 100
    upper = integer + 100
    return [lower,upper]

def perms():
    with open('tests\\permutation_index.pkl', 'rb') as mp:
            index_to_permutation = pkl.load(mp)
            print('Permutation Index Dictionary Loaded')
    return index_to_permutation

def max_index():
    
    permutations = perms()

    oracle = tf_bind_8_oracle()
    
    index = random.sample(range(len(permutations)),100)
        
    initial_sequences = {i:permutations[i] for i in index}

    initial_sequences_int = {index:[string_to_list_int(list_int)] for index, list_int in initial_sequences.items()}

    sequence_preds = {index:oracle.predict(list_int) for index, list_int in initial_sequences_int.items()}

    max_index = max(sequence_preds, key=sequence_preds.get)

    return max_index



class MCMCSequenceSampler:
    def __init__(self, s, l, n):
        self.s = s
        self.l = l
        self.n = n
        self.perms = perms()  
        self.oracle = tf_bind_8_oracle()
        self.index = max_index()
        self.burnin = 1000

    def sample(self):

        all_sequences = []

        burn_in_counter = 0

        while len(all_sequences) <= self.n:

            mu, sigma = self.index, 100

            s = np.random.normal(mu, sigma, 100)

            samples = [int(sample.round()) for sample in s]

            for sample in samples:
                if sample < 0:
                    samples[samples.index(sample)] = 0
                if sample > len(self.perms):
                    samples[samples.index(sample)] = len(samples)

            initial_sequences = {i:self.perms[i] for i in samples}

            initial_sequences_int = {index:[string_to_list_int(list_int)] for index, list_int in initial_sequences.items()}

            sequence_preds = {index:self.oracle.predict(list_int) for index, list_int in initial_sequences_int.items()}

            p = max(sequence_preds, key=sequence_preds.get)

            if string_to_list_int(self.perms[p]) not in all_sequences:
                
                p_likelihood = norm.pdf(p, mu, sigma)
                c_likelihood = norm.cdf(self.index, mu, sigma)

                acceptance_crit = p_likelihood / c_likelihood

                random_number = random.random()

                if random_number < acceptance_crit:

                    self.index = p

                    burn_in_counter += 1

                    print(burn_in_counter)

                    if burn_in_counter > self.burnin:
                        
                        all_sequences.append(string_to_list_int(self.perms[self.index]))

        return all_sequences    
        
        

if __name__ == "__main__":
    s = {'A':0, 'C':1, 'G':2, 'T':3}
    l = 8
    n = 128
    sampler = MCMCSequenceSampler(s, l, n)
    samples = sampler.sample()
    print(samples)