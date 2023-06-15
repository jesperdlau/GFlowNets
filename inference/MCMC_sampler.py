import random
from random_sampler import SequenceSampler
from scipy.stats import gamma
import torch

def target_distribution(x): 
        return torch.sum(torch.exp(-0.5 * x**2))

class MCMCSequenceSampler:
    def __init__(self, burnin, a=0.5, scale=0.5):
        self.burnin = burnin
        self.random_sampler = SequenceSampler()
        self.sequence = self.random_sampler.sample_onehot(1)
        self.a = a
        self.scale = scale
        self.length = 8
        self.alphabet = 4
        
    def sample(self, n):

        all_sequences = []

        burn_in_counter = 0

        while len(all_sequences) <= n-1:
        
            new_sequence = self.sequence.clone().detach()

            changes = 1

            a = self.a
            scale = self.scale

            changes += int(gamma.rvs(a, scale))

            for _ in range(changes):
            
                random_sequence_point = random.randint(0,self.length-1)
                random_sequence_amino = random.randint(0,self.alphabet-1)

                for i in range(self.alphabet):
                    if new_sequence[0][(random_sequence_point*self.alphabet)+i] == 1.:
                        new_sequence[0][(random_sequence_point*self.alphabet)+i] = 0.

                new_sequence[0][random_sequence_point*self.alphabet + random_sequence_amino] = 1.

            p_current = target_distribution(self.sequence[0])
            p_new = target_distribution(new_sequence[0])

            random_number = random.random()
            
            acceptance_crit = min(1.0, p_new / p_current)

            if acceptance_crit > random_number and burn_in_counter < self.burnin:

                burn_in_counter += 1

                self.sequence = new_sequence
            
            elif acceptance_crit > random_number and burn_in_counter >= self.burnin:

                all_sequences.append(new_sequence[0])

                self.sequence = new_sequence

        return torch.stack(all_sequences, dim=0)    
        
class MCMCSequenceSamplerGFP:
    def __init__(self, burnin, a=0.5, scale=0.5):
        self.burnin = burnin
        self.random_sampler = SequenceSampler()
        self.sequence = self.random_sampler.sample_onehot_gfp(1)
        self.a = a
        self.scale = scale
        self.length = 237
        self.alphabet = 20

    def sample(self, n):

        all_sequences = []

        burn_in_counter = 0

        while len(all_sequences) <= n-1:
        
            new_sequence = self.sequence.clone().detach()

            changes = 1

            a = self.a
            scale = self.scale

            changes += int(gamma.rvs(a, scale))

            for _ in range(changes):
            
                random_sequence_point = random.randint(0,self.length-1)
                random_sequence_amino = random.randint(0,self.alphabet-1)

                for i in range(self.alphabet):
                    if new_sequence[0][(random_sequence_point*self.alphabet)+i] == 1.:
                        new_sequence[0][(random_sequence_point*self.alphabet)+i] = 0.

                new_sequence[0][random_sequence_point*self.alphabet + random_sequence_amino] = 1.

            p_current = target_distribution(self.sequence[0])
            p_new = target_distribution(new_sequence[0])

            random_number = random.random()
            
            acceptance_crit = min(1.0, p_new / p_current)

            if acceptance_crit > random_number and burn_in_counter < self.burnin:

                burn_in_counter += 1

                self.sequence = new_sequence
            
            elif acceptance_crit > random_number and burn_in_counter >= self.burnin:

                all_sequences.append(new_sequence[0])

                self.sequence = new_sequence

        return torch.stack(all_sequences, dim=0)


if __name__ == "__main__":
    pass