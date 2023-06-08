import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GFlowNet(nn.Module):
    def __init__(self, num_hid):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(32, num_hid),
                            nn.LeakyReLU(),
                            nn.Linear(num_hid, 4))
        self.keys = ['A', 'C', 'G', 'T'] # Potential discrepency between this vocabular and the source?

    def seq_to_one_hot(self, sequence):
        one_hot = torch.zeros(32, dtype=torch.float)
        for i, letter in enumerate(sequence):
            action = self.keys.index(letter) 
            one_hot[(4*i + action)] = 1.
        return one_hot
    
    def step(self, i, state, action):
        next_state = state.clone() # If not .clone, raises "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:"
        next_state[(4*i + action)] = 1.
        return next_state
    
    def forward(self, x):
        F = self.mlp(x).exp()
        return F
    
    def sample(self):
        pass
    
if __name__ == "__main__":
    model = GFlowNet(512)

    seq = "ACGT"
    x = model.seq_to_one_hot(seq)
    print(f"{x=}")
    x2 = model.step(5, x, 2)
    print(f"{x2=}")

    seq2 = ""
    x3 = model.seq_to_one_hot(seq2)
    print(f"{x3=}")
    x4 = model.step(0, x3, 2)
    print(f"{x4=}")
    


    print()
