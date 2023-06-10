from evaluation.metrics import diversity_par, novelty, performance, novelty_torch

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
        return perf, div, nov

if __name__ == "__main__":
    pass
    #evaluate_modelsampling(X_train, X_sampled, y_sampled, print_stats=True)