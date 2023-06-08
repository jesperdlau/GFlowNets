import random
import numpy as np

class SequenceSampler:
    def __init__(self):
        self.s = {'A':0, 'C':1, 'G':2, 'T':3}
        self.l = 8

    def sample(self, n):
        return np.array([random.choices(list(self.s.values()), k=self.l) for _ in range(n)]).reshape(-1,self.l)

if __name__ == "__main__":
    n = 128
    sampler = SequenceSampler()
    samples = sampler.sample(n)
    print(samples)
