from torch.distributions.categorical import Categorical
import torch

def train(model, opt, reward_func, seq_len, num_episodes, update_freq, device):
    sampled_sequences = []
    terminal_rewards = []
    total_trajectory_flow = []

    losses = []
    minibatch_loss = 0

    reward_func.to(device)
    model.to(device)
    model.train()

    for episode in range(num_episodes):
        # Initialize empty state (as one-hot) and trajectory flow
        state = torch.zeros(32, dtype=torch.float)
        trajectory_flow = []

        # Predict edge flow from initial state
        state.to(device)
        edge_flow_prediction = model(state)

        for i in range(seq_len):
            # Get policy in the current state
            policy = edge_flow_prediction / edge_flow_prediction.sum()

            # Sample action from policy
            action = Categorical(probs=policy).sample() 
            #action.to(device) # TODO: Is it necessary to send action to device?

            # Take action and get new state
            #new_state = state + [model.keys[action]]
            new_state = model.step(i, state, action)
            new_state.to(device)

            # Get the flow from old state to new state
            parent_edge_flow_pred = edge_flow_prediction[action]
            trajectory_flow.append(parent_edge_flow_pred)
        
            # While building the sequence, reward is zero and flow is predicted
            if i < seq_len: 
                reward = 0
                edge_flow_prediction = model(new_state)
                
            # When sequence is complete, get reward and set flow prediction to zero
            else:
                #reward = reward_func(new_state)
                #reward = torch.tensor(reward_func(seq_to_one_hot(new_state))).float()[0] 
                reward = reward_func(new_state) # TODO: Make sure this works
                terminal_rewards.append(reward)
                edge_flow_prediction = torch.zeros(4)
                
            # Calculate the error
            flow_mismatch = (parent_edge_flow_pred - edge_flow_prediction.sum() - reward).pow(2)
            minibatch_loss += flow_mismatch  # Accumulate

            # Continue iterating
            state = new_state

        total_trajectory_flow.append(sum(trajectory_flow))
        sampled_sequences.append(state) # TODO: Possibly go from one-hot to id/char?

        # Perform training step
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
