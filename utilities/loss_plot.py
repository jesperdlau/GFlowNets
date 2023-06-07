import numpy as np
import matplotlib.pyplot as plt
import torch



def loss_plot(path, save_path):
    state_dict = torch.load(path)
    losses = state_dict["losses"]

    plt.plot(losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.savefig(save_path)
    print("Saved loss plot to", save_path)
    