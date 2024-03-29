import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

# Distance measure between two sequences
def distance(tensor1,tensor2):
    # return torch.mean(abs(tensor1 - tensor2))
    return (torch.sum(abs(tensor1-tensor2)))/2

    # pdist = torch.nn.PairwiseDistance(p=2) 
    # return pdist(tensor1,tensor2)

def distance_pair(pair):
    return distance(*pair)

def performance(y_sampled):
    return torch.mean(y_sampled)

# Tæller alle par to gange
# def diversity(X):
#     result = 0
#     for i in range(len(X)):
#         for j in range(len(X)):
#             if i == j:
#                 continue
#             result += distance(X[i],X[j])

#     result /= ((len(X) * (len(X)-1)))

#     return result

def novelty(X_sampled, X_0):
    result = 0
    for i in tqdm(range(len(X_sampled))):
        distances = [distance(X_sampled[i], x) for x in X_0]    
        result += min(distances)
    
    result /= len(X_sampled)

    return result

# optimized functions:
def diversity_par(X_sampled):
    n = len(X_sampled)
    total_distance = 0

    with ThreadPoolExecutor() as executor:
        distances = list(tqdm(executor.map(distance_pair, combinations(X_sampled, 2)), total=n*(n-1)//2))

    total_distance = sum(distances)
    result = total_distance / (n * (n - 1))
    return result

def diversity(X_sampled):
    comb = combinations(X_sampled,2)
    
    sum_ = sum([distance(*pair) for pair in comb])

    result = sum_ / ((len(X_sampled) * (len(X_sampled) - 1)))

    return result

# ikke hurtigere 
def novelty2(X_sampled, X_0):
    result = 0
    for i in tqdm(range(len(X_sampled))):
        min_distance = float('inf')
        
        for x in X_0:
            d = distance(X_sampled[i], x)
            if d < min_distance:
                min_distance = d
        
        result += min_distance
    
    result /= len(X_sampled)

    return result

def novelty_torch(X_sampled, X_0, device = "cpu"):
    dist_m = torch.cdist(X_sampled.to(device), X_0.to(device), p=1) # https://pytorch.org/docs/stable/generated/torch.cdist.html
    min_values = torch.min(dist_m, dim=1).values
    values_sum = torch.sum(min_values)
    result = values_sum / len(X_sampled)
    return result

def novelty_torch_p2(X_sampled, X_0, device = "cpu"):
    dist_m = torch.cdist(X_sampled.to(device), X_0.to(device), p=2) # https://pytorch.org/docs/stable/generated/torch.cdist.html
    min_values = torch.min(dist_m, dim=1).values
    values_sum = torch.sum(min_values)
    result = values_sum / len(X_sampled)
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
        else "cpu")
    print(f"\nUsing {device} device")

    X_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt")
    y_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_train.pt")
    X_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_test.pt")
    y_test  = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_test.pt")

    # p = performance(y_test[:100])
    
    # d = diversity2(X_test[:100])
    # d_par = diversity_par(X_test[:100])
    #n = novelty(X_test[:100],X_train)
    #print(n)
    nov_torch = novelty_torch(X_test[:100],X_train)
    print(f"{nov_torch=}")

    #n2 = novelty2(X_test[:100],X_train)
    # print(f"Performance = {p}")
    # print(f"Diversity = {d_par}")
    # print(f"Diversity = {d}")
    #print(f"Novelty = {n}")
    #print(f"Novelty2 = {n2}")


