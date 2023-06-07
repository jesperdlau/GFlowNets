from torch.distributions.categorical import Categorical
import torch
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def train(model, optimizer, reward_func, seq_len = 8, num_episodes = 100, update_freq = 4, device = "cpu", path = None, hot_start = False):
    """
    Trains a given model using policy gradient with a given reward function.

    Args:
    - model: a PyTorch model used for training
    - optimizer: a PyTorch optimizer used for training
    - reward_func: a function that calculates the reward for a given state
    - seq_len: an int representing the length of each sequence
    - num_episodes: an int representing the number of episodes to train for
    - update_freq: an int representing the frequency of updating the model
    - device: a PyTorch device to use for training
    - path: a string representing the path to save the model to
    - hot_start: a bool indicating whether to load the model from a checkpoint

    Returns:
    - None
    """
    sampled_sequences = []
    terminal_rewards = []
    total_trajectory_flow = []

    losses = []
    minibatch_loss = 0
    start_episode = 0

    # If hot_start is true, load model and optimizer from checkpoint
    if hot_start == True:
        try:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode']
            losses = checkpoint['losses']
        except:
            print("Could not load checkpoint. Starting from scratch.")

    # Move model and reward function to device
    reward_func.to(device)
    model.to(device)
    model.train()

    for episode in range(start_episode, start_episode + num_episodes):
        # Initialize empty state (as one-hot) and trajectory flow
        state = torch.zeros(32, dtype=torch.float, device=device)
        trajectory_flow = []

        # Predict edge flow from initial state
        edge_flow_prediction = model(state)

        for i in range(seq_len):
            # Get policy in the current state
            policy = edge_flow_prediction / edge_flow_prediction.sum()

            # Sample action from policy
            action = Categorical(probs=policy).sample() 
            # action.to(device) # TODO: Is it necessary to send action to device?

            # Take action and get new state
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
                reward = reward_func(new_state) 
                terminal_rewards.append(reward)
                edge_flow_prediction = torch.zeros(4)
                
            # Calculate the error
            flow_mismatch = (parent_edge_flow_pred - edge_flow_prediction.sum() - reward).pow(2)
            minibatch_loss += flow_mismatch  # Accumulate

            # Continue iterating
            state = new_state

        total_trajectory_flow.append(sum(trajectory_flow))
        sampled_sequences.append(state) # TODO: Possibly go from one-hot to id/char? (And save both one-hot and chars..?)

        # print(f"{episode=}, {minibatch_loss.item()=:.2f}")

        # Perform training step
        if episode % update_freq == 0:
            optimizer.zero_grad()
            minibatch_loss.backward()
            optimizer.step()
            
            # Update losses and reset minibatch_loss
            losses.append(minibatch_loss.item())
            minibatch_loss = 0
            # print(f"Performed optimization step")

    # Save checkpoint
    if path:
        torch.save({
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'latest_minibatch_loss': minibatch_loss,
                    'losses': losses
                    }, path)
        print(f"Saved model to {path}")


