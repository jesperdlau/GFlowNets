import torch
import numpy as np
from sklearn.model_selection import train_test_split

DATA_FOLDER = "GFlowNets/data/"

# Rasmus path
DATA_FOLDER = "data/"

SEED = 42
TRAIN_SIZE = 4/5

X = np.load(DATA_FOLDER + "gfp/gfp-x.npy")
y = np.load(DATA_FOLDER + "gfp/gfp-y.npy")

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, shuffle=True, random_state=SEED)

# to tensor
X = torch.tensor(X, dtype=torch.int64)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
X_train = torch.tensor(X_train, dtype=torch.int64)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.int64)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# 1hot encoding
X_train = torch.nn.functional.one_hot(X_train,20).flatten(start_dim=1)
X_test  = torch.nn.functional.one_hot(X_test,20).flatten(start_dim=1)
X       = torch.nn.functional.one_hot(X,20).flatten(start_dim=1)

# correct types
X_train = X_train.type("torch.FloatTensor")
X_test  = X_test.type("torch.FloatTensor")
X       = X.type("torch.FloatTensor")

torch.save(X,"data/gfp/gfp_X.pt")
torch.save(y,"data/gfp/gfp_y.pt")
torch.save(X_train,"data/gfp/gfp_X_train.pt")
torch.save(y_train,"data/gfp/gfp_y_train.pt")
torch.save(X_test,"data/gfp/gfp_X_test.pt")
torch.save(y_test,"data/gfp/gfp_y_test.pt")

