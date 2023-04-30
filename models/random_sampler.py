import random
import numpy as np

class SequenceSampler:
    def __init__(self, s, l, n):
        self.s = s
        self.l = l
        self.n = n

    def sample(self):
        return np.array([random.choices(list(self.s.values()), k=self.l) for _ in range(self.n)]).reshape(-1,self.l)

if __name__ == "__main__":
    s = {'A':0, 'C':1, 'G':2, 'T':3}
    l = 8
    n = 128
    sampler = SequenceSampler(s, l, n)
    samples = sampler.sample()
    print(samples)
