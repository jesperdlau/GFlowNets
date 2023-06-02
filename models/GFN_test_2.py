import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import tqdm


class GFlowNet(nn.Module):
    def __init__(self, num_hid):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(40, num_hid),
                            nn.LeakyReLU(),
                            nn.Linear(num_hid, 5))
        self.keys = ['X', 'A', 'C', 'G', 'T']

    def seq_to_one_hot(self, sequence):
        if len(sequence) > 0:
            token = [self.keys.index(letter) for letter in sequence]
            token = np.pad(token, pad_width=(0, (8-len(sequence))), mode='constant', constant_values=[0])
            token = torch.tensor(token)
            token = F.one_hot(token, num_classes=5).flatten()
            token = token.float()
        else:
            token = torch.zeros(40).float()
        return token
    
    def forward(self, x):
        # F = self.mlp(x).exp() * (1 - x)
        F = self.mlp(x).exp()
        return F
    
model = GFlowNet(512)
opt = torch.optim.Adam(model.parameters(), 3e-4)

losses = []
sampled_sequences = []

minibatch_loss = 0
update_freq = 4

for episode in tqdm.tqdm(range(50), ncols=40):
    state = []

    # Predict F(s, a)
    edge_flow_prediction = model(model.seq_to_one_hot(state))


    for i in range(8):
        policy = edge_flow_prediction / edge_flow_prediction.sum()
        # Sample the action
        action = Categorical(probs=policy).sample() 
        # "Go" to the next state
        new_state = state + [model.keys[action]]

        parent_edge_flow_pred = edge_flow_prediction[action]
    
        if i == 8:
            reward = reward_func(new_state)
            edge_flow_prediction = torch.zeros(5)

        else:
            reward = 0
            edge_flow_prediction = model(model.seq_to_one_hot(new_state))
    
        flow_mismatch = (parent_edge_flow_pred - edge_flow_prediction.sum() - reward).pow(2)
        minibatch_loss += flow_mismatch  # Accumulate
        # Continue iterating
        state = new_state

    sampled_sequences.append(state)
    if episode % update_freq == 0:
        losses.append(minibatch_loss.item())
        minibatch_loss.backward()
        opt.step()
        opt.zero_grad()
        minibatch_loss = 0

class TBModel(nn.Module):
  def __init__(self, num_hid):
    super().__init__()
    # The input dimension is 6 for the 6 patches.
    self.mlp = nn.Sequential(nn.Linear(40, num_hid), nn.LeakyReLU(),
                             # We now output 10 numbers, 5 for P_F and 5 for P_B
                             nn.Linear(num_hid, 10))
    # log Z is just a single number
    self.logZ = nn.Parameter(torch.ones(1))

  def forward(self, x):
    logits = self.mlp(x)
    # Slice the logits, and mask invalid actions (since we're predicting 
    # log-values), we use -100 since exp(-100) is tiny, but we don't want -inf)
    P_F = logits[..., :5] + x * -100
    P_B = logits[..., 5:] * x * -100
    return P_F, P_B