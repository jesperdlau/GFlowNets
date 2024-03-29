import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GFlowNet(nn.Module):
    def __init__(self, n_hid = 2048, n_hidden_layers = 2, n_actions = 4, len_sequence = 8, delta = 0.001):
        super().__init__()
        
        self.n_actions    = n_actions
        self.len_sequence = len_sequence
        self.len_onehot   = self.n_actions * self.len_sequence
        self.n_hid        = n_hid
        self.n_hidden_layers = n_hidden_layers
        self.delta        = delta
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_layer   = nn.Linear(self.len_onehot, self.n_hid)
        output_layer  = nn.Linear(self.n_hid, self.n_actions)
        act_func      = nn.ReLU()
        
        hidden_layers = []
        for _ in range(self.n_hidden_layers):
            hidden_layers.append(nn.Linear(self.n_hid, self.n_hid))
            hidden_layers.append(act_func)

        model_architecture = [input_layer, act_func, *hidden_layers, output_layer]
        self.mlp = nn.Sequential(*model_architecture)
    
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
                action_distribution = torch.mul(action_distribution, (1-self.delta))
                action_distribution = torch.add(action_distribution, self.delta / self.n_actions)
                # action = np.random.choice(self.n_actions, p=action_distribution.detach().numpy())
                action = torch.distributions.Categorical(probs=action_distribution).sample()
                sequence = self.step(i, sequence, action)
            
            sequences[seqN] = sequence
            
        return torch.stack(sequences, dim = 0)
        
if __name__ == "__main__":
    model = GFlowNet(512)
    
    samples = model.sample(500)

    print()
