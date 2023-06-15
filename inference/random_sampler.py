import random
import numpy as np
import torch

class SequenceSampler:
    def __init__(self):
        self.s = {'A':0, 'C':1, 'G':2, 'T':3}
        self.l = 8
        self.alpha = 20
        self.length = 237
        self.state = self.alpha * self.length

    def sample(self, n):
        return np.array([random.choices(list(self.s.values()), k=self.l) for _ in range(n)]).reshape(-1,self.l)
    
    def sample_onehot(self, n):
        samples = []
        for _ in range(n):
            one_hot = torch.zeros(32, dtype=torch.float)
            for i in range(8):
                action = np.random.randint(0, 4)
                one_hot[(4*i + action)] = 1.
            samples.append(one_hot)
        return torch.stack(samples, dim=0)

    def sample_onehot_gfp(self, n):
        samples = []
        for _ in range(n):
            one_hot = torch.zeros(self.state, dtype=torch.float)
            for i in range(self.length):
                action = np.random.randint(0, 20)
                one_hot[(self.alpha*i + action)] = 1.
            samples.append(one_hot)
        return torch.stack(samples, dim=0)
        
    

if __name__ == "__main__":
    n = 128
    sampler = SequenceSampler()
    
    # one = sampler.sample_onehot(4)
    # print(one)
    # print()

    gfp = sampler.sample_onehot_gfp(4)
    print(gfp)
