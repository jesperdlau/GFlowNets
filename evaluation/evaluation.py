from evaluation.metrics import diversity_par, novelty, performance, novelty_torch
import torch
import numpy as np

def evaluate_modelsampling(X_train,X_sampled,y_sampled, print_stats = True):
    """
    Evaluates the performance, diversity, and novelty of a sampled dataset and returns the results.

    Args:
        X_train (torch.tensor): The original dataset.
        X_sampled (torch.tensor): The sampled dataset.
        y_sampled (torch.tensor): The labels for the sampled dataset.
        print_stats (bool): Whether to print the statistics or not. Default is True.

    Returns:
        Tuple of three floats, representing the performance, diversity, and novelty of the sampled dataset.

    Example:
        evaluate_modelsampling(X_train, X_sampled, y_sampled, print_stats=True) returns (0.95, 7.5, 3.5)
    """
        
    perf = performance(y_sampled)
    div  = diversity_par(X_sampled)
    nov  = novelty_torch(X_sampled, X_train)

    if print_stats:
        print(f"Performance: {perf}")
        print(f"Diversity: {div}")
        print(f"Novelty: {nov}")
    
    else:
        return perf.cpu(), div.cpu(), nov.cpu()
    
def get_top20percent(X, y):
    mask = torch.argsort(y, dim=0, descending=True)
    X_sorted = X[mask].squeeze()
    y_sorted = y[mask].squeeze()
    X_top20 = X_sorted[:int(len(y) * 0.2)]
    y_top20 = y_sorted[:int(len(y) * 0.2)]
    return X_top20, y_top20

def evaluate_batches(X_sampled, y_sampled, X_train, print_stats = False):
    metrics_list = []
    for X, y in zip(X_sampled, y_sampled):
        X_top20, y_top20 = get_top20percent(X, y)
        perf, div, nov = evaluate_modelsampling(X_train, X_top20, y_top20, print_stats=print_stats)
        metrics_list.append({"Performance": perf.detach().numpy().item(), 
                             "Diversity": div.detach().numpy().item(), 
                             "Novelty": nov.detach().numpy().item()})
    return np.array(metrics_list)


if __name__ == "__main__":
    pass
    #evaluate_modelsampling(X_train, X_sampled, y_sampled, print_stats=True)