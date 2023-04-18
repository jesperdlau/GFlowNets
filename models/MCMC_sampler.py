import random

class MCMCSequenceSampler:
    def __init__(self, s, l, n):
        self.s = s
        self.l = l
        self.n = n

    def sample(self):
        # Initialize the current sequence to a random sequence
        current_sequence = ''.join(random.choices(self.s, k=self.l))
        
        # Generate n sequences using MCMC sampling
        sequences = []
        for i in range(self.n):
            sequences.append(current_sequence)
            
            # Generate a new candidate sequence by randomly changing a single character in the current sequence
            candidate_sequence = list(current_sequence)
            j = random.randint(0, self.l-1)
            candidate_sequence[j] = random.choice(self.s)
            candidate_sequence = ''.join(candidate_sequence)
            
            # Calculate the acceptance probability, will always be 1/l or 0.0, due to only changing one letter with the possibility of being the same or not
            p_accept = min(1, sum([1 for c1, c2 in zip(current_sequence, candidate_sequence) if c1 != c2])/self.l)
            
            # Decide whether to accept the candidate sequence
            if random.random() < p_accept:
                current_sequence = candidate_sequence
        
        return sequences

if __name__ == "__main__":
    s = 'ACGT'
    l = 8
    n = 128
    sampler = MCMCSequenceSampler(s, l, n)
    samples = sampler.sample()
    print(samples)