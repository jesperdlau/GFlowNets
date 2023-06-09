import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import tqdm
import matplotlib.pyplot as plt
from tf_bind_8_reward import TFBindReward1HOT
import pickle

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

with open("permutation_values.pkl", "rb") as fp:
        permutation_values = pickle.load(fp)
        print('Perm values')
        print(permutation_values)

total_reward = sum(permutation_values.values())

# Load reward function
reward_func = TFBindReward1HOT()
reward_path = "data/tf_bind_8/SIX6_REF_R1/TFBind_1hot_test.pth"
reward_func.load_state_dict(torch.load(reward_path))

class TBModel(nn.Module):
  def __init__(self, num_hid):
    super().__init__()
    # The input dimension is 6 for the 6 patches.
    self.mlp = nn.Sequential(nn.Linear(32, num_hid), nn.LeakyReLU(),
                             # We now output 10 numbers, 5 for P_F and 5 for P_B
                             nn.Linear(num_hid, 8))
    # log Z is just a single number
    self.logZ = nn.Parameter(torch.ones(1))
    self.keys = ['A', 'C', 'G', 'T'] # Potential discrepency between this vocabular and the source?

    
  def forward(self, x):
    logits = self.mlp(x) # TODO: should it be .exp()?
    # Slice the logits, and mask invalid actions (since we're predicting 
    # log-values), we use -100 since exp(-100) is tiny, but we don't want -inf)
    P_F = logits[..., :4] * -100
    P_B = logits[..., 4:] * -100
    return P_F, P_B

  def seq_to_one_hot(self, sequence):
        if len(sequence) > 0:
            token = [self.keys.index(letter) + 1 for letter in sequence] # +1 Off-set so 0 is no-character
            token = np.pad(token, pad_width=(0, (8-len(sequence))), mode='constant', constant_values=[0])
            token = torch.tensor(token)
            token = F.one_hot(token.to(torch.int64), num_classes=5).flatten()
            token = token.float()
        else:
            token = torch.zeros(32).float()
        return token  

model = TBModel(512)
opt = torch.optim.Adam(model.parameters(),  3e-4)

print(f"\nUsing {device} device")

# Let's keep track of the losses and the faces we sample
tb_losses = []
tb_sampled_sequences = []
# To not complicate the code, I'll just accumulate losses here and take a 
# gradient step every `update_freq` episode.
minibatch_loss = 0
update_freq = 2

logZs = []
log_reward_fractions = []
reward_fractions = []
log_backward_probabilities = []
delta_reward_probs = []
partition_total_reward_delta = []

for episode in tqdm.tqdm(range(1000), ncols=40):
  # Each episode starts with an "empty state"
  state = []
  # Predict P_F, P_B
  P_F_s, P_B_s = model(model.seq_to_one_hot(state))
  total_P_F = 0
  total_P_B = 0

  for t in range(8):
    # Here P_F is logits, so we want the Categorical to compute the softmax for us
    cat = Categorical(logits=P_F_s)
    action = cat.sample() 
    # "Go" to the next state
    new_state = state + [model.keys[action]]
    # Accumulate the P_F sum
    total_P_F += cat.log_prob(action)

    if t == 7: 
      # If we've built a complete face, we're done, so the reward is > 0
      # (unless the face is invalid)
      reward = torch.tensor(reward_func(model.seq_to_one_hot(new_state))).float()
    # We recompute P_F and P_B for new_state
    P_F_s, P_B_s = model(model.seq_to_one_hot(new_state))
    # Here we accumulate P_B, going backwards from `new_state`. We're also just 
    # going to use opposite semantics for the backward policy. I.e., for P_F action
    # `i` just added the face part `i`, for P_B we'll assume action `i` removes 
    # face part `i`, this way we can just keep the same indices. 
    total_P_B += Categorical(logits=P_B_s).log_prob(action)

    # Continue iterating
    state = new_state

  # We're done with the trajectory, let's compute its loss. Since the reward can
  # sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
  loss = (model.logZ + total_P_F - torch.log(reward).clip(-20) - total_P_B).pow(2)
  minibatch_loss += loss

  #Testflow: 

  log_reward_fraction = torch.log(reward) / torch.log(torch.tensor(total_reward))
  reward_fraction = reward / total_reward

  proportionality = torch.abs(total_P_B - log_reward_fraction)

  delta_reward_probs.append(proportionality)
  partition_total_reward_delta.append(torch.abs(model.logZ.exp() - total_reward))

  # Add the face to the list, and if we are at an
  # update episode, take a gradient step.
  tb_sampled_sequences.append(state)
  if episode % update_freq == 0:
    tb_losses.append(minibatch_loss.item())
    minibatch_loss.backward()
    opt.step()
    opt.zero_grad()
    minibatch_loss = 0
    logZs.append(model.logZ.item())

f, ax = plt.subplots(2, 1, figsize=(10,6))
plt.sca(ax[0])
plt.plot(tb_losses)
plt.yscale('log')
plt.ylabel('loss')
plt.sca(ax[1])
plt.plot(np.exp(logZs))
plt.ylabel('estimated Z')

plt.show()

f, ax = plt.subplots(2, 1, figsize=(10,6))
plt.sca(ax[0])
plt.plot([element.item() for element in delta_reward_probs])
plt.ylabel('Reward Probability Proportionality')
plt.sca(ax[1])
plt.plot([element.item() for element in partition_total_reward_delta])
plt.ylabel('Total flow Total reward Delta')

plt.show()

print([element.item() for element in delta_reward_probs])
print([element.item() for element in partition_total_reward_delta])

print(model.logZ.exp())


