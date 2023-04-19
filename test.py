# Test

import numpy as np
# import scipy.stats as st
# import matplotlib.pyplot as plt

# data1 = np.linspace(0, 1, 100)
# data2 = np.linspace(10, 11, 100)
# data = np.concatenate((data1, data2))


#create 95% confidence interval for population mean weight
# interval = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
# mean = np.mean(data)

# print(interval)
# print(mean)


# def test():
#     print()


import design_bench

import design_bench
task = design_bench.make('TFBind8-Exact-v0')

def solve_optimization_problem(x0, y0):
    return x0  # solve a model-based optimization problem

# solve for the best input x_star and evaluate it
x_star = solve_optimization_problem(task.x, task.y)
y_star = task.predict(x_star)

import numpy as np

a = np.array([2,3])
print(a)
