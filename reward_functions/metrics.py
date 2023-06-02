import torch
import numpy as np
from tqdm import tqdm

# Distance measure between two sequences
def distance(seq1,seq2):
    return torch.mean(abs(seq1 - seq2))

def performance(y):
    return torch.mean(y)

def diversity(X):
    result = 0
    for i in tqdm(range(len(X))):
        for j in range(len(X)):
            if i == j:
                continue
            result += distance(X[i],X[j])

    result /= (len(X) * (len(X)-1))

    return result

def novelty(X_new, X_0):
    result = 0
    for i in tqdm(range(len(X_new))):
        distances = [distance(X_new[i], x) for x in X_0]    
        result += min(distances)
    
    result /= len(X_new)

    return result


if __name__ == "__main__":
    DATA_FOLDER = "GFlowNets/data/"

    # Rasmus path
    DATA_FOLDER = "data/"

    SEED = 42

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device")   

    X_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt")
    y_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_train.pt")
    X_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_test.pt")
    y_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_test.pt")

    p = performance(y_test[:100])
    d = diversity(X_test[:100])
    n = novelty(X_test[:100],X_train)
    
    print(f"Performance = {p}")
    print(f"Diversity = {d}")
    print(f"Novelty = {n}")



