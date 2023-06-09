import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GFlowNet(nn.Module):
    def __init__(self, n_hid, n_hidden_layers = 0):
        super().__init__()
        
        self.keys         = ['A', 'C', 'G', 'T'] # Potential discrepency between this vocabular and the source?
        self.n_actions    = 4
        self.len_sequence = 8
        self.len_onehot   = self.n_actions * self.len_sequence
        self.n_hid        = n_hid
        self.n_hidden_layers = n_hidden_layers

        input_layer   = nn.Linear(self.len_onehot, self.n_hid)
        output_layer  = nn.Linear(self.n_hid, self.n_actions)
        act_func      = nn.LeakyReLU()
        
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
        F = self.mlp(x).exp()
        return F
    
    def sample(self, N=1):
        sequences = [None for _ in range(N)]

        for seqN in range(N):
            sequence = torch.zeros(self.len_onehot, dtype=torch.float)
            
            for i in range(self.len_sequence):
                edge_flow_prediction = self.forward(sequence)
                action_distribution = edge_flow_prediction / torch.sum(edge_flow_prediction) 
                # action = np.random.choice(self.n_actions, p=action_distribution.detach().numpy())
                action = torch.distributions.Categorical(probs=action_distribution).sample()
                sequence = self.step(i, sequence, action)
            
            sequences[seqN] = sequence
            
        return torch.stack(sequences, dim = 0)
        
if __name__ == "__main__":
    model = GFlowNet(512)
    
    samples = model.sample(500)

    print()
