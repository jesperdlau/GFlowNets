import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GFlowNet(nn.Module):
    def __init__(self, n_hid = 2048, n_hidden_layers = 2, n_actions = 4, len_sequence = 8):
        super().__init__()
        
        self.keys         = ['A', 'C', 'G', 'T'] # Potential discrepency between this vocabular and the source?
        self.n_actions    = n_actions
        self.len_sequence = len_sequence
        self.len_onehot   = self.n_actions * self.len_sequence
        self.n_hid        = n_hid
        self.n_hidden_layers = n_hidden_layers
        self.logZ = nn.Parameter(torch.ones(1))

        input_layer   = nn.Linear(self.len_onehot, self.n_hid)
        output_layer  = nn.Linear(self.n_hid, self.n_actions*2)
        act_func      = nn.ReLU()
        
        hidden_layers = []
        for _ in range(self.n_hidden_layers):
            hidden_layers.append(nn.Linear(self.n_hid, self.n_hid))
            hidden_layers.append(act_func)

        model_architecture = [input_layer, act_func, *hidden_layers, output_layer]
        self.mlp = nn.Sequential(*model_architecture)
       
    def seq_to_one_hot(self, sequence):
        one_hot = torch.zeros(self.len_onehot, dtype=torch.float)
        for i, letter in enumerate(sequence):
            action = self.keys.index(letter) 
            one_hot[(self.n_actions * i + action)] = 1.
        return one_hot
    
    def step(self, i, state, action):
        next_state = state.clone() # If not .clone, raises "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:"
        next_state[(self.n_actions * i + action)] = 1.
        return next_state
    
    def forward(self, x):
        F = self.mlp(x)
        P_F = F[...,:self.n_actions]
        P_B = F[...,self.n_actions:]
        return P_F, P_B
    
    def sample(self, N=1):
        sequences = [None for _ in range(N)]

        for seqN in range(N):
            sequence = torch.zeros(self.len_onehot, dtype=torch.float)
            
            for i in range(self.len_sequence):
                P_F_s, _ = self.forward(sequence)
                action = torch.distributions.Categorical(logits=P_F_s).sample()
                sequence = self.step(i, sequence, action)
            
            sequences[seqN] = sequence
            
        return torch.stack(sequences, dim = 0)
        
if __name__ == "__main__":
    model = GFlowNet(512)
    
    samples = model.sample(500)

    print(samples)
