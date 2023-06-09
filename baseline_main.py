import torch
from models.random_sampler import SequenceSampler
from utilities.transformer import Transformer
from evaluation.metrics import diversity, performance, diversity_par
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
import time
from evaluation.evaluation import evaluate_modelsampling

DATA_FOLDER = "data/"

# Load reward function
reward_func = TFBindReward1HOT()
reward_path = "models/saved_models/TFBind_1hot_test.pth"
reward_func.load_state_dict(torch.load(reward_path))

alphabet = ['A', 'C', 'G', 'T']

s = {'A':0, 'C':1, 'G':2, 'T':3}
l = 8
n = 128

random_sampler = SequenceSampler()

trans = Transformer(alphabet)

random_sampler = SequenceSampler()
#MCMC_sampler = MCMCSequenceSampler(2)


random_samples = random_sampler.sample(100)
#MCMC_samples = MCMC_sampler.sample(SAMPLE_SIZE)
#gflow_samples = model.sample(SAMPLE_SIZE)

random_samples = trans.list_list_int_to_tensor_one_hot(random_samples)
#MCMC_samples = helper.list_list_int_to_tensor_one_hot(MCMC_samples)
#gflow_samples = helper.list_list_int_to_tensor_one_hot(gflow_samples)

#predict

random_reward = reward_func(random_samples)

X_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt")

random_evaluation = evaluate_modelsampling(X_train,random_samples,random_reward, print_stats = True)