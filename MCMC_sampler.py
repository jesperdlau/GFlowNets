import random
import numpy as np
import pickle as pkl
from tf_bind_8_oracle import tf_bind_8_oracle
from scipy.stats import norm
from models.random_sampler import SequenceSampler
from scipy.stats import gamma
import torch

PERMUTATION_PATH = "tests/permutation_index.pkl"

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
    # with open('tests\\permutation_index.pkl', 'rb') as mp: # Only windows? 
    with open(PERMUTATION_PATH, 'rb') as mp:
            index_to_permutation = pkl.load(mp)
            #print('Permutation Index Dictionary Loaded')
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
    def __init__(self, burnin):
        self.burnin = burnin
        self.random_sampler = SequenceSampler()
        self.sequence = self.random_sampler.sample_onehot(1)
        self.length = 8
        self.alphabet = 4
        

    def sample(self, n):

        all_sequences = []

        burn_in_counter = 0

        while len(all_sequences) <= n-1:
        
            new_sequence = self.sequence.clone().detach()
            
            random_sequence_point = random.randint(0,self.length-1)
            random_sequence_amino = random.randint(0,self.alphabet-1)

            for i in range(self.alphabet):
                if new_sequence[0][(random_sequence_point*self.alphabet)+i] == 1.:
                    new_sequence[0][(random_sequence_point*self.alphabet)+i] = 0.

            new_sequence[0][random_sequence_point*self.alphabet + random_sequence_amino] = 1.

            changes = 1

            for i in range(self.alphabet*self.length):
                if new_sequence[0][i] == 1.:
                    if new_sequence[0][i] != self.sequence[0][i]:
                        changes += 1

            a = 0.5
            scale = 0.5

            p_current = gamma.pdf(changes, a, scale)
            p_new = gamma.pdf(changes+1, a, scale)

            random_number = random.random()
            
            acceptance_crit = p_new / p_current

            if acceptance_crit > random_number and burn_in_counter < self.burnin:

                burn_in_counter += 1

                self.sequence = new_sequence
            
            elif acceptance_crit > random_number and burn_in_counter >= self.burnin:

                all_sequences.append(new_sequence[0])

                self.sequence = new_sequence

        return torch.stack(all_sequences, dim=0)    
        
class MCMCSequenceSamplerGFP:
    def __init__(self, burnin):
        self.burnin = burnin
        self.random_sampler = SequenceSampler()
        self.sequence = self.random_sampler.sample_onehot_gfp(1)
        self.length = 237
        self.alphabet = 20

    def sample(self, n):

        all_sequences = []

        burn_in_counter = 0

        while len(all_sequences) <= n-1:
        
            new_sequence = self.sequence.clone().detach()
            
            random_sequence_point = random.randint(0,self.length-1)
            random_sequence_amino = random.randint(0,self.alphabet-1)

            for i in range(self.alphabet):
                if new_sequence[0][(random_sequence_point*self.alphabet)+i] == 1.:
                    new_sequence[0][(random_sequence_point*self.alphabet)+i] = 0.

            new_sequence[0][random_sequence_point*self.alphabet + random_sequence_amino] = 1.

            changes = 1

            for i in range(self.alphabet*self.length):
                if new_sequence[0][i] == 1.:
                    if new_sequence[0][i] != self.sequence[0][i]:
                        changes += 1

            a = 0.5
            scale = 0.5

            p_current = gamma.pdf(changes, a, scale)
            p_new = gamma.pdf(changes+1, a, scale)

            random_number = random.random()
            
            acceptance_crit = p_new / p_current

            if acceptance_crit > random_number and burn_in_counter < self.burnin:

                burn_in_counter += 1

                self.sequence = new_sequence
            
            elif acceptance_crit > random_number and burn_in_counter >= self.burnin:

                all_sequences.append(new_sequence[0])

                self.sequence = new_sequence

        return torch.stack(all_sequences, dim=0)


if __name__ == "__main__":
    n = 128
    burnin = 1
    sampler = MCMCSequenceSamplerGFP(burnin)
    tf_sampler = MCMCSequenceSampler(burnin)

    tf_samples = tf_sampler.sample(20)
    samples = sampler.sample(20)
    
    print(tf_samples)
    print(samples)