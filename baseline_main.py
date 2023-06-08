import torch
from models.random_sampler import SequenceSampler
from utilities.transformer import transformer
from evaluation.metrics import diversity, performance, diversity_par
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
import time

# Load reward function
reward_func = TFBindReward1HOT()
reward_path = "models/saved_models/TFBind_1hot_test.pth"
reward_func.load_state_dict(torch.load(reward_path))

alphabet = ['A', 'C', 'G', 'T']

s = {'A':0, 'C':1, 'G':2, 'T':3}
l = 8
n = 128

random_sampler = SequenceSampler(s, l, n)

trans = transformer(alphabet)

samples = random_sampler.sample()

one_hot_samples = trans.list_list_int_to_tensor_one_hot(samples)

reward_func.eval()
with torch.no_grad():
    y = torch.tensor([reward_func(x) for x in one_hot_samples])

diverse_metric = diversity(one_hot_samples)
diverse_par_metric = diversity_par(one_hot_samples)

initial = time.time()
print(diverse_metric)
final = time.time()
print('diverse: ',final - initial)

initial = time.time()
print(diverse_par_metric)
final = time.time()
print('diverse_par: ',final - initial)
print(performance(y))