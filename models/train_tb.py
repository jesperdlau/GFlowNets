from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import time
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


# TODO: Slet ikke færdig, bare kopiret fra den anden train funktion
def train_tb(model, optimizer, logz_optimizer, reward_func, 
             minibatch_size:int = 32, num_episodes:int = 100, checkpoint_freq:int = 50,
             delta:float = 0.001, beta:int = 3,
             model_path = None, reward_path = None, device = "cpu", hot_start:bool = False, verbose:bool = False):
    """
    Trains a given model using policy gradient with a given reward function.

    Args:
    - model: a PyTorch GFlow model used for training
    - optimizer: a PyTorch optimizer used for training
    - reward_func: a function that calculates the reward for a given state
    - num_episodes: an int representing the number of episodes to train for
    - update_freq: an int representing the frequency of updating the model
    - delta: a float representing the exploration rate
    - beta: int representing the exponent of reward
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
    # terminal_rewards = []
    # total_trajectory_flow = []

    # logZ = nn.Parameter(torch.ones(1))

    # Move model and reward function to device
    reward_func.to(device)
    reward_func.load_state_dict(torch.load(reward_path, map_location=device))

    model.to(device)
    model.train()

    # If hot_start is true, load model and optimizer from checkpoint
    # if hot_start: # TODO: Needs to be modified if needed
    #     try:
    #         checkpoint = torch.load(model_path)
    #         model_state_dict = checkpoint['model_state_dict']
    #         model.load_state_dict(model_state_dict)
    #         optimizer_state_dict = checkpoint['optimizer_state_dict']
    #         optimizer.load_state_dict(optimizer_state_dict)
    #         start_episode = checkpoint['episode'] 
    #         minibatch_loss = checkpoint['minibatch_loss']
    #         losses = checkpoint['losses']
    #     except:
    #         print("Could not load checkpoint. Starting from scratch.")
    
    # Setup dictionary for saving checkpoints during run
    checkpoint = {
                'checkpoint_step': [],
                'model_state_dict': [],
                'optimizer_state_dict': [],
                'logZ': [],
                'logz_optimizer_state_dict': [],
                'minibatch_loss': [],
                'average_loss': [],
                'n_hid': model.n_hid,
                'n_hidden_layers': model.n_hidden_layers
                }

    # Training outer loop (5000)
    start_time = time.time()
    for training_step in range(num_episodes):    
        minibatch_loss = torch.zeros(1)

        # Training inner loop (32)
        for episode in range(minibatch_size):
            # Initialize empty state (as one-hot) and trajectory flow
            state = torch.zeros(model.len_onehot, dtype=torch.float, device=device)
            state = state.to(device)

            # Predict edge flow from initial state
            P_F_s, P_B_s  = model(state)
            total_P_F = 0
            total_P_B = 0

            # Build sequence
            for i in range(model.len_sequence):
                # Get policy in the current state
                policy = P_F_s

                # Adding uniform distribution to policy, delta controls exploration
                policy = torch.mul(policy, (1-delta))
                policy = torch.add(policy, delta * (1/model.n_actions))   

                # Sample action from policy
                distribution = Categorical(logits=policy)
                action = distribution.sample()

                total_P_F += distribution.log_prob(action)
                # action.to(device) # TODO: Is it necessary to send action to device?
                new_state = model.step(i, state, action)
                new_state.to(device)

                if i == model.len_sequence - 1:
                    reward = reward_func(new_state).pow(beta)

                P_F_s, P_B_s = model(new_state)
                # Take action and get new state
                total_P_B += Categorical(logits=P_B_s).log_prob(action)
                
                # Continue iterating
                state = new_state

            reward = torch.nan_to_num(torch.log(reward).clip(-20), -20) # clips reward to at least -20, even if it is nan
            loss = (model.logZ + total_P_F - reward - total_P_B).pow(2)
            minibatch_loss += loss.cpu()
            
            # total_trajectory_flow.append(sum(trajectory_flow))
            sampled_sequences.append(state) # TODO: Possibly go from one-hot to id/char? (And save both one-hot and chars..?)

        # Print log
        if verbose:
            print(f"Step:[{training_step+1:>4}]/[{num_episodes}] \t Time:[{time.time() - start_time:>8.0f}s] \t Loss:[{minibatch_loss.item():>14.8f}]")
        # Perform optimization
        optimizer.zero_grad()
        logz_optimizer.zero_grad()

        minibatch_loss.backward()

        optimizer.step()
        logz_optimizer.step()
        #print("Performed optimization")

        # Average loss for step
        average_step_loss = minibatch_loss.item() / minibatch_size

        # Save checkpoint
        if model_path and (training_step+1) % checkpoint_freq == 0:
            model_state_dict = model.state_dict().copy()
            model_state_dict_cpu = {k: v.cpu() for k, v in model_state_dict.items()}
            optimizer_state_dict = optimizer.state_dict().copy()
            logZ = model.logZ.item()
            logz_optimizer_state_dict = logz_optimizer.state_dict().copy()

            checkpoint['checkpoint_step'].append(training_step)
            checkpoint['model_state_dict'].append(model_state_dict_cpu)
            checkpoint['optimizer_state_dict'].append(optimizer_state_dict)
            checkpoint['logZ'].append(logZ)
            checkpoint['logz_optimizer_state_dict'].append(logz_optimizer_state_dict)
            checkpoint['minibatch_loss'].append(minibatch_loss.item())
            checkpoint['average_loss'].append(average_step_loss)

            torch.save(checkpoint, model_path)
            print(f"Saved model checkpoint")

        # Reset minibatch_loss
        minibatch_loss = torch.zeros(1)


