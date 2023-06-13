import torch
import numpy as np
from sklearn.model_selection import train_test_split

DATA_FOLDER = "GFlowNets/data/"

# Rasmus path
DATA_FOLDER = "data/"

SEED = 42
TRAIN_SIZE = 1/2
VALIDATION_SIZE = 0.1

X = np.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_8-x.npy")
y = np.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_8-y.npy")

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, shuffle=True, random_state=SEED)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=VALIDATION_SIZE, shuffle=True, random_state=SEED)

# to tensor
X = torch.tensor(X, dtype=torch.int64)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
X_train = torch.tensor(X_train, dtype=torch.int64)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_valid = torch.tensor(X_valid, dtype=torch.int64)
y_valid = torch.tensor(y_valid, dtype=torch.float32).reshape(-1, 1)

# 1hot encoding
X_train = torch.nn.functional.one_hot(X_train,4).flatten(start_dim=1)
X_valid  = torch.nn.functional.one_hot(X_valid,4).flatten(start_dim=1)
X       = torch.nn.functional.one_hot(X,4).flatten(start_dim=1)

# correct types
X_train = X_train.type("torch.FloatTensor")
X_valid  = X_valid.type("torch.FloatTensor")
X       = X.type("torch.FloatTensor")

# torch.save(X,"data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X.pt")
# torch.save(y,"data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y.pt")
# torch.save(X_train,"data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt")
# torch.save(y_train,"data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_train.pt")
# torch.save(X_test,"data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_test.pt")
# torch.save(y_test,"data/tf_bind_8/SIX6_REF_R1/tf_bind_1hot_y_test.pt")

# Save
torch.save(X,"data/tfbind8/tfbind8_X.pt")
torch.save(y,"data/tfbind8/tfbind8_y.pt")
torch.save(X_train,"data/tfbind8/tfbind8_X_train.pt")
torch.save(y_train,"data/tfbind8/tfbind8_y_train.pt")
torch.save(X_valid,"data/tfbind8/tfbind8_X_valid.pt")
torch.save(y_valid,"data/tfbind8/tfbind8_y_valid.pt")

print("1-hot preprocessing complete")
