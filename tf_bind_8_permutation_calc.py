from tf_bind_8_oracle import tf_bind_8_oracle
import numpy as np
 
# Create all the permutations
def perms(arr, Len, L): 
    # There can be (Len)^l permutations
    full = []
    for n in range(pow(Len, L)):
        s = []
        for i in range(L):
            
            # Print the ith element
            # of sequence
            s.append(arr[n % Len])
            n //= Len
        full.append(s)
    return np.array(full).reshape(-1,L)
    
if __name__ == "__main__":
    arr = [0, 1, 2, 3]
    Len = len(arr)
    L = 8
    
    # function call
    permutations = perms(arr, Len, L)
    print(permutations)
    np.save('tf_bind_8_permutations', permutations)

    # code for loading
    '''
    loaded = np.load('tf_bind_8_permutations.npy')
    print(loaded)
    '''