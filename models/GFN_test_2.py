import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import tqdm
import matplotlib.pyplot as plt


from tf_bind_8_reward import TFBindReward1HOT

# Load reward function
reward_func = TFBindReward1HOT()
reward_path = "data/tf_bind_8/SIX6_REF_R1/TFBind_1hot_test.pth"
reward_func.load_state_dict(torch.load(reward_path))

# 
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

terminal_rewards = []
total_trajectory_flow = []

for episode in tqdm.tqdm(range(1000), ncols=40):
    state = []

    # Predict F(s, a)
    edge_flow_prediction = model(model.seq_to_one_hot(state))

    trajectory_flow = []

    for i in range(8):
        policy = edge_flow_prediction / edge_flow_prediction.sum()
        # Sample the action
        action = Categorical(probs=policy).sample() 
        # "Go" to the next state
        new_state = state + [model.keys[action]]

        parent_edge_flow_pred = edge_flow_prediction[action]

        trajectory_flow.append(parent_edge_flow_pred)
    
        if i == 7: # changed from 8
            reward = torch.tensor(reward_func(model.seq_to_one_hot(new_state))).float()[0] #Reward to tensor rep index 1
            terminal_rewards.append(reward)
            edge_flow_prediction = torch.zeros(5)

        else:
            reward = 0
            edge_flow_prediction = model(model.seq_to_one_hot(new_state))
    
        flow_mismatch = (parent_edge_flow_pred - edge_flow_prediction.sum() - reward).pow(2)
        minibatch_loss += flow_mismatch  # Accumulate
        # Continue iterating
        state = new_state

    total_trajectory_flow.append(sum(trajectory_flow))

    sampled_sequences.append(state)
    if episode % update_freq == 0:
        losses.append(minibatch_loss.item())
        minibatch_loss.backward()
        opt.step()
        opt.zero_grad()
        minibatch_loss = 0

# Plot
[print(seq) for seq in sampled_sequences[:10]]

plt.figure(figsize=(10,3))
plt.plot(losses)
plt.ylabel('Loss')
plt.yscale('log')
plt.show()

plt.figure(figsize=(10,3))
plt.plot([element.item() for element in total_trajectory_flow])
plt.ylabel('Total Trajectory Flow for full sequences n=8')
plt.show()

delta_trajectory_flow_reward = [abs(flow.item() - reward.item()) for flow, reward in zip(total_trajectory_flow,terminal_rewards)]

plt.figure(figsize=(10,3))
plt.plot(delta_trajectory_flow_reward)
plt.ylabel('Delta Total flow of Trajectory vs. Reward of Trajectory')
plt.show()

print(delta_trajectory_flow_reward[-10:])


