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


def performance_plot(losses, random_perf, gflow_perf, save_path):
    x_axis = np.arange(1, len(losses)+1, 1)
    fig, ax1 = plt.subplots()

    color = 'black'
    ax1.set_xlabel('Minibatch')
    ax1.set_ylabel('Log-Loss', color=color)
    ax1.set_yscale("log")
    ax1.plot(x_axis, losses, color=color, label="Log-Loss", linestyle="--")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Performance', color="blue")  # we already handled the x-label with ax1
    ax2.plot(x_axis, gflow_perf, label="GFlow", color="red")

    ax2.axhline(y=random_perf, color="blue", label="Random")

    ax2.tick_params(axis='y', labelcolor="blue")
    plt.xticks(x_axis, x_axis)
    plt.xlabel("Minibatch")
    plt.title("Log-Loss and Performance Over Minibatches")
    fig.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_path)
    plt.close()
    print("Saved Performance plot to", save_path)

def diversity_plot(losses, random_diversity, gflow_diversity, save_path):
    x_axis = np.arange(1, len(losses)+1, 1)
    fig, ax1 = plt.subplots()

    color = 'black'
    ax1.set_xlabel('Minibatch')
    ax1.set_ylabel('Log-Loss', color=color)
    ax1.set_yscale("log")
    ax1.plot(x_axis, losses, color=color, label="Log-Loss", linestyle="--")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Diversity', color="blue")  # we already handled the x-label with ax1
    ax2.plot(x_axis, gflow_diversity, label="GFlow", color="red")

    ax2.axhline(y=random_diversity, color="blue", label="Random")

    ax2.tick_params(axis='y', labelcolor="blue")
    plt.xticks(x_axis, x_axis)
    plt.xlabel("Minibatch")
    plt.title("Log-Loss and Diversity Over Minibatches")
    fig.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_path)
    plt.close()
    print("Saved diversity plot to", save_path)

def novelty_plot(losses, random_novel, gflow_novel, save_path):
    x_axis = np.arange(1, len(losses)+1, 1)
    fig, ax1 = plt.subplots()

    color = 'black'
    ax1.set_xlabel('Minibatch')
    ax1.set_ylabel('Log-Loss', color=color)
    ax1.set_yscale("log")
    ax1.plot(x_axis, losses, color=color, label="Log-Loss", linestyle="--")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Novelty', color="blue")  # we already handled the x-label with ax1
    ax2.plot(x_axis, gflow_novel, label="GFlow", color="red")

    ax2.axhline(y=random_novel, color="blue", label="Random")

    ax2.tick_params(axis='y', labelcolor="blue")
    plt.xticks(x_axis, x_axis)
    plt.xlabel("Minibatch")
    plt.title("Log-Loss and Novelty Over Minibatches")
    fig.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_path)
    plt.close()
    print("Saved novelty plot to", save_path)




def combined_plot(losses, random_mean, random_ci, mcmc_mean, mcmc_ci, gflow_data, save_path, plot_type:str):
    x_axis = np.arange(1, len(losses)+1, 1)
    fig, ax1 = plt.subplots()

    color = 'black'
    ax1.set_ylabel('Log-Loss', color=color)
    ax1.set_yscale("log")
    ax1.plot(x_axis, losses, color=color, label="Log-Loss", linestyle="--")
    #ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    ax2.plot(x_axis, gflow_data, label="GFlow", color="red")
    ax2.axhline(y=random_mean, color="blue", label="Random")
    ax2.axhline(y=mcmc_mean, color="blue", label="MCMC")
    ax2.fill_between(x_axis, mcmc_ci[0], mcmc_ci[1], color="blue", alpha=0.2)
    ax2.fill_between(x_axis, random_ci[0], random_ci[1], color="blue", alpha=0.2)

    ax2.set_ylabel(ylabel=plot_type, color="blue")  # we already handled the x-label with ax1
    #ax2.tick_params(axis='y', labelcolor="blue")



    #plt.xticks(x_axis, x_axis)
    plt.xlabel("Batch")
    plt.title("Log-Loss and Novelty Over Minibatches")
    fig.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_path)
    plt.close()
    print("Saved combined plot to", save_path)


