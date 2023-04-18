import matplotlib.pyplot as pp
import numpy as np

current_sequence = ''
l = 8

s = {
    'A': 'Adenine',
    'C': 'Cytosine',
    'G': 'Guanine',
    'T': 'Thymine'
}

sorted_keys = sorted(s.keys())

def reward(sequence):
    
    reward = 0

    if len(sequence) <= 1:
        return reward

    length = len(sequence)

    if len(sequence) % 2 != 0:
        length = len(sequence) - 1
    
    even_numbers = [i for i in range(0, length, 2)]

    for number in even_numbers:
        if sequence[number] == 'A' and sequence[number+1] == 'T':
            reward += 0.2
        if sequence[number] == 'T' and sequence[number+1] == 'A':
            reward += 0.2
        if sequence[number] == 'G' and sequence[number+1] == 'C':
            reward += 0.2
        if sequence[number] == 'C' and sequence[number+1] == 'G':
            reward += 0.2

    return reward


