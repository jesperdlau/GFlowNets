from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)



# TODO: Slet ikke færdig, bare kopiret fra den anden train funktion
def train_tb(model, optimizer, reward_func, seq_len = 8, num_episodes = 100, update_freq = 4, model_path = None, reward_path = None, device = "cpu", hot_start = False, verbose = False):
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
    - model_path: a string representing the path to save the model to, or load from if hot_start
    - reward_path: a string representing the path to load the reward function from
    - device: the device where the model, optimizer, and reward function are sent to (default is "cpu")
    - hot_start: a bool indicating whether to load the model from a checkpoint
    - verbose: a bool indicating whether to print info about each episode

    Returns:
    - None
    """
    sampled_sequences = []
    terminal_rewards = []
    total_trajectory_flow = []

    losses = []
    minibatch_loss = torch.zeros(1)
    start_episode = 0
    logZ = nn.Parameter(torch.ones(1))

    # Move model and reward function to device
    reward_func.to(device)
    reward_func.load_state_dict(torch.load(reward_path, map_location=device))

    model.to(device)
    model.train()

    # If hot_start is true, load model and optimizer from checkpoint
    if hot_start:
        try:
            checkpoint = torch.load(model_path)
            model_state_dict = checkpoint['model_state_dict']
            model.load_state_dict(model_state_dict)
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            optimizer.load_state_dict(optimizer_state_dict)
            start_episode = checkpoint['episode'] 
            minibatch_loss = checkpoint['minibatch_loss']
            losses = checkpoint['losses']
        except:
            print("Could not load checkpoint. Starting from scratch.")


    for episode in range(start_episode + 1, start_episode + num_episodes + 1):
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
            minibatch_loss += flow_mismatch.cpu()  # Accumulate
            # Continue iterating
            state = new_state

        total_trajectory_flow.append(sum(trajectory_flow))
        sampled_sequences.append(state) # TODO: Possibly go from one-hot to id/char? (And save both one-hot and chars..?)

        if verbose:
            print(f"{episode=},\t {minibatch_loss.item()=:.2f}")

        # Perform training step
        if episode % update_freq == 0 and episode != 0:
            optimizer.zero_grad()
            minibatch_loss.backward()
            optimizer.step()
            
            # Update losses and reset minibatch_loss
            losses.append(minibatch_loss.item())
            minibatch_loss = torch.zeros(1)
            if verbose:
                print(f"Performed optimization step")

    # Save checkpoint
    if model_path:
        model_state_dict = model.state_dict()
        model_state_dict_cpu = {k: v.cpu() for k, v in model_state_dict.items()}
        optimizer_state_dict = optimizer.state_dict()
        torch.save({
                    'episode': episode,
                    'model_state_dict': model_state_dict_cpu,
                    'optimizer_state_dict': optimizer_state_dict,
                    'minibatch_loss': minibatch_loss.item(), # Is this necessary?
                    'losses': losses
                    }, model_path)
        print(f"Saved model to {model_path}")


