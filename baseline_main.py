import torch
from inference.random_sampler import SequenceSampler
from utilities.transformer import Transformer
from evaluation.metrics import diversity, performance, diversity_par
from reward_functions.tf_bind_reward_1hot import TFBindReward1HOT
import time
from evaluation.evaluation import evaluate_modelsampling
from inference.MCMC_sampler import MCMCSequenceSampler
from inference.MCMC_light_sampler import MCMCLightSequenceSampler

DATA_FOLDER = "data/"

# Load reward function
reward_func = TFBindReward1HOT()
reward_path = "models/saved_models/TFBind_1hot_test.pth"
reward_func.load_state_dict(torch.load(reward_path))

alphabet = ['A', 'C', 'G', 'T']

s = {'A':0, 'C':1, 'G':2, 'T':3}
l = 8
n = 128
burnin = [1,10,100,1000,10000]
std_dev = [0.1,1,10,100,1000]
SAMPLE_SIZE = 100

'''

random_sampler = SequenceSampler()
MCMC_light = MCMCLightSequenceSampler(burnin, std_dev)
MCMC_sampler = MCMCSequenceSampler(burnin, std_dev)

helper = Transformer(alphabet)

MCMC_light_sampler = MCMCLightSequenceSampler(burnin, std_dev)
MCMC_sampler = MCMCSequenceSampler(burnin, std_dev)


random_samples = random_sampler.sample(SAMPLE_SIZE)
MCMC_light_samples = MCMC_light_sampler.sample(SAMPLE_SIZE)
MCMC_samples = MCMC_sampler.sample(SAMPLE_SIZE)
#gflow_samples = model.sample(SAMPLE_SIZE)

random_samples = helper.list_list_int_to_tensor_one_hot(random_samples)
MCMC_light_samples = helper.list_list_int_to_tensor_one_hot(MCMC_light_samples)
MCMC_samples = helper.list_list_int_to_tensor_one_hot(MCMC_samples)
#gflow_samples = helper.list_list_int_to_tensor_one_hot(gflow_samples)

#predict

random_reward = reward_func(random_samples)
MCMC_light_reward = reward_func(MCMC_light_samples)
MCMC_reward = reward_func(MCMC_samples)

X_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt")

random_evaluation = evaluate_modelsampling(X_train,random_samples,random_reward, print_stats = True)
MCMC_light_evaluation = evaluate_modelsampling(X_train,MCMC_light_samples,MCMC_light_reward, print_stats = True)
MCMC_evaluation = evaluate_modelsampling(X_train,MCMC_samples,MCMC_reward, print_stats = True)

'''

if __name__ == "__main__":
    DATA_FOLDER = "data/"

    # Load reward function
    reward_func = TFBindReward1HOT()
    reward_path = "models/saved_models/TFBind_1hot_test.pth"
    reward_func.load_state_dict(torch.load(reward_path))

    alphabet = ['A', 'C', 'G', 'T']

    s = {'A':0, 'C':1, 'G':2, 'T':3}
    l = 8
    n = 128
    burnin = [1,10,100,1000,10000]
    std_dev_light = [1,5,10,50,100]
    std_dev = [10,100,1000,5000,10000]
    SAMPLE_SIZE = 10

    log = {burn:{std:[] for std in std_dev} for burn in burnin}
    light_log = {burn:{std:[] for std in std_dev_light} for burn in burnin}
    
    for burn in burnin: 
        for std, std_light in zip(std_dev, std_dev_light):

            MCMC_light_sampler = MCMCLightSequenceSampler(burn, std_light)
            MCMC_sampler = MCMCSequenceSampler(burn, std)

            helper = Transformer(alphabet)

            MCMC_light_samples = MCMC_light_sampler.sample(SAMPLE_SIZE)
            MCMC_samples = MCMC_sampler.sample(SAMPLE_SIZE)
            #gflow_samples = model.sample(SAMPLE_SIZE)

            MCMC_light_samples = helper.list_list_int_to_tensor_one_hot(MCMC_light_samples)
            MCMC_samples = helper.list_list_int_to_tensor_one_hot(MCMC_samples)
            #gflow_samples = helper.list_list_int_to_tensor_one_hot(gflow_samples)

            #predict

            MCMC_light_reward = reward_func(MCMC_light_samples)
            MCMC_reward = reward_func(MCMC_samples)

            X_train = torch.load(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_1hot_X_train.pt")

            MCMC_light_evaluation = evaluate_modelsampling(X_train,MCMC_light_samples,MCMC_light_reward, print_stats = True)
            MCMC_evaluation = evaluate_modelsampling(X_train,MCMC_samples,MCMC_reward, print_stats = True)

            log[burn][std].append(MCMC_evaluation)
            light_log[burn][std_light].append(MCMC_light_evaluation)
    print(log)
    print(light_log)

    '''

    Initial grid search w. 10 samples:

        burnin = [1,10,100,1000,10000]
        std_dev_light = [1,5,10,50,100]
        std_dev = [10,100,1000,5000,10000]

    for burn, std in log.items(): print(burn, std)
        1 {10: [(tensor(0.5183, grad_fn=<MeanBackward0>), tensor(2.8889), tensor(0.6000))], 100: [(tensor(0.8839, grad_fn=<MeanBackward0>), tensor(3.1111), tensor(0.4000))], 1000: [(tensor(0.5758, grad_fn=<MeanBackward0>), tensor(4.4889), tensor(0.8000))], 5000: [(tensor(0.5402, grad_fn=<MeanBackward0>), tensor(5.2667), tensor(0.4000))], 10000: [(tensor(0.4752, grad_fn=<MeanBackward0>), tensor(5.8000), tensor(0.4000))]}
        10 {10: [(tensor(0.4467, grad_fn=<MeanBackward0>), tensor(2.7778), tensor(0.8000))], 100: [(tensor(0.5740, grad_fn=<MeanBackward0>), tensor(3.7111), tensor(0.2000))], 1000: [(tensor(0.5982, grad_fn=<MeanBackward0>), tensor(4.2667), tensor(0.4000))], 5000: [(tensor(0.6158, grad_fn=<MeanBackward0>), tensor(5.2000), tensor(0.))], 10000: [(tensor(0.6300, grad_fn=<MeanBackward0>), tensor(5.1556), tensor(0.2000))]}
        100 {10: [(tensor(0.5453, grad_fn=<MeanBackward0>), tensor(1.9556), tensor(0.2000))], 100: [(tensor(0.6994, grad_fn=<MeanBackward0>), tensor(3.5111), tensor(0.))], 1000: [(tensor(0.5102, grad_fn=<MeanBackward0>), tensor(5.0444), tensor(0.))], 5000: [(tensor(0.5107, grad_fn=<MeanBackward0>), tensor(5.8444), tensor(0.4000))], 10000: [(tensor(0.5149, grad_fn=<MeanBackward0>), tensor(5.4444), tensor(0.4000))]}
        1000 {10: [(tensor(0.5659, grad_fn=<MeanBackward0>), tensor(1.9778), tensor(0.8000))], 100: [(tensor(0.7453, grad_fn=<MeanBackward0>), tensor(2.8889), tensor(0.4000))], 1000: [(tensor(0.5772, grad_fn=<MeanBackward0>), tensor(5.2889), tensor(0.2000))], 5000: [(tensor(0.5297, grad_fn=<MeanBackward0>), tensor(4.6444), tensor(0.4000))], 10000: [(tensor(0.5291, grad_fn=<MeanBackward0>), tensor(5.6444), tensor(0.4000))]}
        10000 {10: [(tensor(0.8730, grad_fn=<MeanBackward0>), tensor(2.4222), tensor(0.2000))], 100: [(tensor(0.8807, grad_fn=<MeanBackward0>), tensor(3.2667), tensor(0.6000))], 1000: [(tensor(0.5794, grad_fn=<MeanBackward0>), tensor(4.6222), tensor(0.4000))], 5000: [(tensor(0.6078, grad_fn=<MeanBackward0>), tensor(5.2889), tensor(0.4000))], 10000: [(tensor(0.5322, grad_fn=<MeanBackward0>), tensor(5.7556), tensor(0.4000))]}
    
    for burn, std in light_log.items(): print(burn, std)
        1 {1: [(tensor(0.9481, grad_fn=<MeanBackward0>), tensor(1.5556), tensor(0.4000))], 5: [(tensor(0.9012, grad_fn=<MeanBackward0>), tensor(2.0889), tensor(0.2000))], 10: [(tensor(0.5754, grad_fn=<MeanBackward0>), tensor(2.2444), tensor(0.))], 50: [(tensor(0.4328, grad_fn=<MeanBackward0>), tensor(2.6222), tensor(0.6000))], 100: [(tensor(0.5622, grad_fn=<MeanBackward0>), tensor(3.), tensor(0.6000))]}
        10 {1: [(tensor(0.8380, grad_fn=<MeanBackward0>), tensor(1.5333), tensor(0.4000))], 5: [(tensor(0.4719, grad_fn=<MeanBackward0>), tensor(2.), tensor(0.))], 10: [(tensor(0.7386, grad_fn=<MeanBackward0>), tensor(2.1556), tensor(0.4000))], 50: [(tensor(0.4957, grad_fn=<MeanBackward0>), tensor(3.), tensor(0.2000))], 100: [(tensor(0.7366, grad_fn=<MeanBackward0>), tensor(2.9556), tensor(0.4000))]}
        100 {1: [(tensor(0.8018, grad_fn=<MeanBackward0>), tensor(2.6000), tensor(0.2000))], 5: [(tensor(0.4832, grad_fn=<MeanBackward0>), tensor(1.9111), tensor(0.4000))], 10: [(tensor(0.8838, grad_fn=<MeanBackward0>), tensor(1.7556), tensor(0.))], 50: [(tensor(0.6672, grad_fn=<MeanBackward0>), tensor(3.2667), tensor(0.4000))], 100: [(tensor(0.4156, grad_fn=<MeanBackward0>), tensor(3.4000), tensor(0.4000))]}
        1000 {1: [(tensor(0.7857, grad_fn=<MeanBackward0>), tensor(1.5333), tensor(0.4000))], 5: [(tensor(0.4807, grad_fn=<MeanBackward0>), tensor(1.8889), tensor(0.2000))], 10: [(tensor(0.3078, grad_fn=<MeanBackward0>), tensor(2.), tensor(0.))], 50: [(tensor(0.4329, grad_fn=<MeanBackward0>), tensor(3.6000), tensor(0.4000))], 100: [(tensor(0.4948, grad_fn=<MeanBackward0>), tensor(3.6444), tensor(0.4000))]}
        10000 {1: [(tensor(0.3951, grad_fn=<MeanBackward0>), tensor(2.2000), tensor(0.4000))], 5: [(tensor(0.4068, grad_fn=<MeanBackward0>), tensor(2.0667), tensor(0.2000))], 10: [(tensor(0.5923, grad_fn=<MeanBackward0>), tensor(2.2889), tensor(0.4000))], 50: [(tensor(0.4353, grad_fn=<MeanBackward0>), tensor(2.8889), tensor(0.4000))], 100: [(tensor(0.3894, grad_fn=<MeanBackward0>), tensor(3.5333), tensor(0.4000))]}
    '''