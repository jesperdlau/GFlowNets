from tf_bind_8_oracle import tf_bind_8_oracle
import numpy as np
import pickle as pkl
import json

if __name__ == "__main__":
    permutations = np.load('tf_bind_8_permutations.npy')
    print(permutations)

    oracle = tf_bind_8_oracle()

    preds = oracle.predict(permutations)

    print(preds)

    permutation_value_dictionary = {}

    for count, permutation in enumerate(permutations):
        
        string_perm = ''

        for element in permutation:
            string_perm += str(element)
        
        permutation_value_dictionary[count] = string_perm
    
    with open('permutation_index.pkl', 'wb') as fp:
        pkl.dump(permutation_value_dictionary, fp)
        print('succesfully saved permutations!')

    #For loading pickle file:
    '''
    with open('permutation_index.pkl', 'rb') as fp:
        permutation_values = pickle.load(fp)
        print('Perm values')
        print(permutation_values)
    '''
    

