import matplotlib.pyplot as pp
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import tqdm

class FlowModel(nn.Module):
  def __init__(self, num_hid):
    super().__init__()
    # We encoded the current state as binary vector, for each patch the associated 
    # dimension is either 0 or 1 depending on the absence or precense of that patch.
    # Therefore the input dimension is 6 for the 6 patches.
    self.mlp = nn.Sequential(nn.Linear(40, num_hid), nn.LeakyReLU(),
                             # We also output 6 numbers, since there are up to
                             # 6 possible actions (and thus child states), but we 
                             # will mask those outputs for patches that are 
                             # already there.
                             nn.Linear(num_hid, 5))
  def forward(self, x):
    # We take the exponential to get positive numbers, since flows must be positive,
    # and multiply by (1 - x) to give 0 flow to actions we know we can't take
    # (in this case, x[i] is 1 if a feature is already there, so we know we 
    # can't add it again).
    F = self.mlp(x).exp()
    return F

def parent_sequences(state):
    z = torch.zeros(40)
    r = torch.reshape(z,(8,5))
    parent_states = []
    parent_actions = []
    for i in range(len(state)-1,0,-1):
        if not torch.all(state[i].eq(torch.zeros(5))):
            parent_actions.append(state[i])
            state[i] = torch.zeros(5)
            parent_states.append(state)
    return parent_states, parent_actions

def return_index(action):
  for i in range(len(action)):
    if action[i] == 1:
      return i

# Instantiate model and optimizer
F_sa = FlowModel(512)
opt = torch.optim.Adam(F_sa.parameters(), 3e-4)

# Let's keep track of the losses and the faces we sample
losses = []
sampled_sequences = []
# To not complicate the code, I'll just accumulate losses here and take a 
# gradient step every `update_freq` episode.
minibatch_loss = 0
update_freq = 4
for episode in tqdm.tqdm(range(50000), ncols=40):
  # Each episode starts with an "empty state"
  state = torch.zeros(40)
  state = torch.reshape(state,(8,5))
  # Predict F(s, a)
  edge_flow_prediction = F_sa(state.flatten())
  print('edge_flow_prediction ',edge_flow_prediction)
  for t in range(8):
    # The policy is just normalizing, and gives us the probability of each action
    policy = edge_flow_prediction / edge_flow_prediction.sum()
    print('policy ',policy)
    # Sample the action
    action = Categorical(probs=policy).sample() 
    print('action ', action)
    # "Go" to the next state
    new_state = state.clone().detach()
    new_state[t][action] = 1
    # Now we want to compute the loss, we'll first enumerate the parents
    # And compute the edge flows F(s, a) of each parent
    parent_flow = F_sa(state)
    parent_edge_flow_preds = parent_flow[action]

    # Now we need to compute the reward and F(s, a) of the current state,
    # which is currently `new_state`
    if t == 7: 
      # If we've built a complete face, we're done, so the reward is > 0
      # (unless the face is invalid)
      reward = face_reward(new_state)
      print('reward', face_reward)
      # and since there are no children to this state F(s,a) = 0 \forall a
      edge_flow_prediction = torch.zeros(6)
      print('edge_flow_prediction ', edge_flow_prediction)
    else:
      # Otherwise we keep going, and compute F(s, a)
      reward = 0
      print('reward ', reward)
      edge_flow_prediction = F_sa(face_to_tensor(new_state))
      print('edge_flow_prediction ', edge_flow_prediction)

    # The loss as per the equation above
    flow_mismatch = (parent_edge_flow_preds.sum() - edge_flow_prediction.sum() - reward).pow(2)
    print('loss (flow_mismatch) ', flow_mismatch)
    minibatch_loss += flow_mismatch  # Accumulate
    print('minibatch_loss ', minibatch_loss)
    # Continue iterating
    state = new_state
    print('state ', state)

  # We're done with the episode, add the face to the list, and if we are at an
  # update episode, take a gradient step.
  sampled_faces.append(state)
  print('sampled_faces ', sampled_faces)
  if episode % update_freq == 0:
    losses.append(minibatch_loss.item())
    print('losses ', losses)
    minibatch_loss.backward()
    print('minibatch_loss',minibatch_loss)
    opt.step()
    opt.zero_grad()
    minibatch_loss = 0



    