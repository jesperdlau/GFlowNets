import torch

s = ['A', 'C', 'G', 'T']

class Transformer:
    def __init__(self, alphabet):

        self.alphabet = alphabet
    
    def list_string_to_list_one_hot(list_string):
        for sequence in list_string:
            for element in sequence:
                print()
    def list_list_string_to_tensor_one_hot(self, list_strings):
        full_tensor = torch.zeros(len(list_strings),32)
        for k,list_string in enumerate(list_strings):
            one_hot = torch.zeros(32, dtype=torch.float)
            for sequence in list_string:
                for i, letter in enumerate(sequence):
                    action = self.alphabet.index(letter) 
                    one_hot[(4*i + action)] = 1.
            full_tensor[k] = one_hot
        return full_tensor

    def list_list_int_to_tensor_one_hot(self, list_list_int):
        full_tensor = torch.zeros(len(list_list_int),32)
        for k, list_int in enumerate(list_list_int):
            one_hot = torch.zeros(32, dtype=torch.float)
            for i, integer in enumerate(list_int):
                one_hot[(4*i + integer)] = 1.
            full_tensor[k] = one_hot
        return full_tensor

if __name__ == "__main__":
    model = Transformer(s)

    print(model.list_list_string_to_tensor_one_hot([['ACT'], ['ACT'], ['AGT']]))
    print(model.list_list_int_to_tensor_one_hot([[2,1,2,3,0,1,2,3], [2,1,2,3,0,1,2,3]]))