import pickle as pkl
import numpy as np


if __name__ == "__main__":
    with open('permutation_values.pkl', 'rb') as fp:
        permutation_values = pkl.load(fp)

    sequence_atoms = [0,1,2,3]

    value_sums = {i:{k:0 for k in range(4)} for i in range(8)}

    #A total of 16.384 items in each bucket
    #value_counts = {i:{k:0 for k in range(4)} for i in range(8)}

    for key in permutation_values:
        for position in range(len(key)):
            for atom in sequence_atoms:
                if key[position] == str(atom):
                    value_sums[position][atom] += permutation_values[key]/16384
                    #value_counts[position][atom] += 1

    sorted_permutation_values = {k: v for k, v in sorted(permutation_values.items(), key=lambda item: item[1])}
    first_2000_pairs = {k: sorted_permutation_values[k] for k in list(sorted_permutation_values)[:2000]}
    last_2000_pairs = {k: sorted_permutation_values[k] for k in list(sorted_permutation_values)[len(sorted_permutation_values)-2000:]}
    with open('tf_bind_8_reward_proportionality_min_set.pkl', 'wb') as lp:
        pkl.dump(first_2000_pairs, lp)
        print('succesfully saved reward proportionalities minimum set !')

    with open('tf_bind_8_reward_proportionality_max_set.pkl', 'wb') as qp:
        pkl.dump(last_2000_pairs, qp)
        print('succesfully saved reward proportionalities maximum set !')



    with open('tf_bind_8_reward_proportionality.pkl', 'wb') as tp:
        pkl.dump(value_sums, tp)
        print('succesfully saved reward proportionalities!')

    #For loading pickle file:
    '''
    with open('tf_bind_8_reward_proportionality.pkl', 'rb') as tp:
        reward_proportionality = pickle.load(tp)
        print('Reward proportionality values')
        print(reward_proportionality)
    '''

