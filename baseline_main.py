import torch
from models.random_sampler import SequenceSampler
from utilities.transformer import transformer
from evaluation.metrics import diversity, performance
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT 

# Load reward function
reward_func = TFBindReward1HOT()
reward_path = "data/tf_bind_8/SIX6_REF_R1/TFBind_1hot_test.pth"
reward_func.load_state_dict(torch.load(reward_path))

alphabet = ['A', 'C', 'G', 'T']

s = {'A':0, 'C':1, 'G':2, 'T':3}
l = 8
n = 128

sampler = SequenceSampler(s, l, n)

trans = transformer(alphabet)

samples = sampler.sample()

one_hot_samples = trans.list_list_int_to_tensor_one_hot(samples)



print(one_hot_samples.shape)