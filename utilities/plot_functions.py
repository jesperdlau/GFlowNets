import numpy as np
import matplotlib.pyplot as plt
import torch

def loss_plot(losses, save_path):
    x_axis = np.arange(1, len(losses)+1, 1)

    plt.plot(x_axis, losses, color="black")

    plt.xticks(x_axis, x_axis)
    plt.yscale("log")
    plt.xlabel("Minibatch")
    plt.ylabel("Log-Loss")
    plt.title("Log-Loss")
    plt.savefig(save_path)
    plt.close()
    print("Saved loss plot to", save_path)
    

def eval_plot(perfs, divs, novels, save_path):
    x_axis = np.arange(1, len(perfs)+1, 1)

    plt.plot(x_axis, perfs, label="Performance")
    plt.plot(x_axis, divs, label="Diversity")
    plt.plot(x_axis, novels, label="Novelty")

    plt.xticks(x_axis, x_axis)
    plt.xlabel("Minibatch")
    plt.ylabel("Evaluation")
    plt.title("Evaluation Metrics")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print("Saved eval plot to", save_path)


def combined_loss_eval_plot(losses, perfs, divs, novels, save_path):
    x_axis = np.arange(1, len(perfs)+1, 1)
    fig, ax1 = plt.subplots()

    color = 'black'
    ax1.set_xlabel('Minibatch')
    ax1.set_ylabel('Log-Loss', color=color)
    ax1.set_yscale("log")
    ax1.plot(x_axis, losses, color=color, label="Log-Loss")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Eval Metrics', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_axis, perfs, label="Performance")
    ax2.plot(x_axis, divs, label="Diversity")
    ax2.plot(x_axis, novels, label="Novelty")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.xticks(x_axis, x_axis)
    plt.xlabel("Minibatch")
    plt.title("Log-Loss and Evaluation Metrics")
    fig.legend()
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("Saved combined loss eval plot to", save_path)


def combined_loss_eval_plot_flex(losses, perfs, divs, novels, save_path):
    x_axis = np.arange(1, len(losses)+1, 1)
    fig, ax1 = plt.subplots()

    color = 'black'
    ax1.set_xlabel('Minibatch')
    ax1.set_ylabel('Log-Loss', color=color)
    ax1.set_yscale("log")
    ax1.plot(x_axis, losses, color=color, label="Log-Loss")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Evaluation', color="blue")  # we already handled the x-label with ax1
    if perfs:
        ax2.plot(x_axis, perfs, label="Performance")
    if divs:
        ax2.plot(x_axis, divs, label="Diversity")
    if novels:
        ax2.plot(x_axis, novels, label="Novelty")
    ax2.tick_params(axis='y', labelcolor="blue")

    plt.xticks(x_axis, x_axis)
    plt.xlabel("Minibatch")
    plt.title("Log-Loss and Evaluation Metrics")
    fig.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_path)
    plt.close()
    print("Saved combined loss eval plot to", save_path)
