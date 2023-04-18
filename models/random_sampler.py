import random

class SequenceSampler:
    def __init__(self, s, l, n):
        self.s = s
        self.l = l
        self.n = n

    def sample(self):
        return [''.join(random.choices(self.s, k=self.l)) for _ in range(self.n)]

if __name__ == "__main__":
    s = 'ACGT'
    l = 8
    n = 128
    sampler = SequenceSampler(s, l, n)
    samples = sampler.sample()
    print(samples)
