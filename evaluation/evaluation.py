from evaluation.metrics import diversity_par, novelty, performance 

def evaluate_modelsampling(X_0,X_sampled,y_sampled, print_stats = True):
    """
    Evaluates the performance, diversity, and novelty of a sampled dataset and returns the results.

    Args:
        X_0 (torch.tensor): The original dataset.
        X_sampled (torch.tensor): The sampled dataset.
        y_sampled (torch.tensor): The labels for the sampled dataset.
        print_stats (bool): Whether to print the statistics or not. Default is True.

    Returns:
        Tuple of three floats, representing the performance, diversity, and novelty of the sampled dataset.

    Example:
        evaluate_modelsampling(X_train, X_sampled, y_sampled, print_stats=True) returns (0.95, 0.85, 0.75)
    """
        
    perf = performance(y_sampled)
    div  = diversity_par(X_sampled)
    nov  = novelty(X_sampled, X_0)

    if print_stats:
        print(f"Performance: {perf}")
        print(f"Diversity: {div}")
        print(f"Novelty: {nov}")
    
    return perf, div, nov

if __name__ == "__main__":
    pass
    #evaluate_modelsampling(X_train, X_sampled, y_sampled, print_stats=True)