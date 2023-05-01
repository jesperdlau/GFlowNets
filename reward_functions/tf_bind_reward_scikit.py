import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader, Dataset
import pickle 

DATA_FOLDER = "GFlowNets/data/"

# Rasmus path
DATA_FOLDER = "data/"

SEED = 42
TRAIN_SIZE = 4/5
EPOCHS = 30
BATCH_SIZE = 100
LEARNING_RATE = 0.0001

X = np.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_8-x.npy")
y = np.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_8-y.npy").reshape()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, shuffle=True, random_state=SEED)

model = MLPRegressor(random_state=SEED,max_iter = 100,verbose=True,tol=0.00001)
model.fit(X_train,y_train)

preds = model.predict(X_test)
score = model.score(X_test,y_test)
print(score)

plt.scatter(preds,y_test)
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.xlim(0,1)
plt.ylim(0,1)
plt.title("Plot of the fitted and observed values")
plt.show()

filename = 'tf_bind_scikit.sav'
pickle.dump(model,open(filename,"wb"))
