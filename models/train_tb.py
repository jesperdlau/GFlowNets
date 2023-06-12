from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


# TODO: Slet ikke færdig, bare kopiret fra den anden train funktion
def train_tb(model, optimizer, reward_func, num_episodes:int = 100, update_freq:int = 4, delta:float = 0.001, beta:int = 3,model_path = None, reward_path = None, device = "cpu", hot_start:bool = False, verbose:bool = False):
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

    losses = []
    minibatch_loss = torch.zeros(1)
    start_episode = 0
    logZ = nn.Parameter(torch.ones(1))

    # Move model and reward function to device
    reward_func.to(device)
    reward_func.load_state_dict(torch.load(reward_path, map_location=device))

    model.to(device)
    model.train()
    models = []

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
        state = torch.zeros(model.len_onehot, dtype=torch.float, device=device)
        trajectory_flow = []

        # Predict edge flow from initial state
        P_F_s, P_B_s  = model(state)
        total_P_F = 0
        total_P_B = 0


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
                reward = reward_func(new_state)

            P_F_s, P_B_s = model(new_state)
            # Take action and get new state
            total_P_B += Categorical(logits=P_B_s).log_prob(action)
            
            # Continue iterating
            state = new_state

        loss = (model.logZ + total_P_F - torch.log(reward).clip(-20) - total_P_B).pow(2)
        minibatch_loss += loss.cpu()
        
        # total_trajectory_flow.append(sum(trajectory_flow))
        sampled_sequences.append(state) # TODO: Possibly go from one-hot to id/char? (And save both one-hot and chars..?)

        if verbose:
            print(f"{episode=},\t {minibatch_loss.item()=:.2f}")

        # Perform training step
        if episode % update_freq == 0 and episode != 0:
            optimizer.zero_grad()
            minibatch_loss.backward()
            optimizer.step()
            '''
            # Update logZ
            logz_optmizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''
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


