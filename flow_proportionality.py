import pickle

with open("permutation_values.pkl", "rb") as fp:
        permutation_values = pickle.load(fp)
        print('Perm values')
        print(permutation_values)

total_reward = sum(permutation_values.values())
print(total_reward)





