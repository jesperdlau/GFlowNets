import numpy as np
import xgboost as xgb
import torch
from torch_helperfunctions import set_device, MinMaxScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


if __name__ == "__main__":
    device = set_device()
    DATA_FOLDER = "data/"
    LEARNING_RATE = 0.3
    EVAL_METRIC = "rmse"
    N_ESTIMATORS = 500
    MAX_DEPTH = 6

    VERBOSE = True
    PERFORM_GRIDSEARCH = False

    X_train = torch.load(DATA_FOLDER + "gfp/gfp_1hot_X_train.pt")
    y_train = torch.load(DATA_FOLDER + "gfp/gfp_1hot_y_train.pt")
    X_test  = torch.load(DATA_FOLDER + "gfp/gfp_1hot_X_test.pt")
    y_test  = torch.load(DATA_FOLDER + "gfp/gfp_1hot_y_test.pt")

    # normalizing y
    y_train = MinMaxScaler(y_train,0,1)
    y_test = MinMaxScaler(y_test,0,1)

    # To numpy
    X_train = X_train.squeeze().numpy()
    y_train = y_train.squeeze().numpy()
    X_test  = X_test.squeeze().numpy()
    y_test  = y_test.squeeze().numpy()

    if PERFORM_GRIDSEARCH:
        param_grid = {"max_depth":    [4, 5, 6],
                "n_estimators": [500, 600, 700],
                "learning_rate": [0.01, 0.015]}
        model = xgb.XGBRegressor(eval_metric='rmse')
        search = GridSearchCV(model, param_grid, cv=5).fit(X_train, y_train)
        N_ESTIMATORS = search.best_params_["n_estimators"]
        MAX_DEPTH    = search.best_params_["max_depth"]

    model = xgb.XGBRegressor(learning_rate = LEARNING_RATE,
                             n_estimators = N_ESTIMATORS,
                             max_depth = MAX_DEPTH,
                             eval_metric = EVAL_METRIC,
                             early_stopping_rounds = 6)
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=VERBOSE)

    preds = model.predict(X_test)

    plt.scatter(preds,y_test)
    plt.xlabel("Predictions")
    plt.ylabel("Labels")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title("Plot of the fitted and observed values")
    plt.show()






