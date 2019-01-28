# Thompson Sampling Reinforced Learning

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

# Import the data
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
FilePath = BinPath + '\Ads_CTR_Optimisation.csv';
dataset = pd.read_csv(FilePath)

# Implamenting Thompson Sampling
N = 10000
d = 10
tot_reward = 0

ID_selected = []
num_rewards_1 = [0] * d
num_rewards_0 = [0] * d

for n in range(0, N):
    ID = 0
    max_rand = 0
    for i in range(0, d):
        rand_beta = random.betavariate(num_rewards_1[i] + 1, num_rewards_0[i] + 1)
        if rand_beta > max_rand:
            max_rand = rand_beta
            ID = i
    ID_selected.append(ID)
    reward = dataset.values[n, ID]
    if(reward == 1):
        num_rewards_1[ID] = num_rewards_1[ID] + 1
    else:
        num_rewards_0[ID] = num_rewards_0[ID] + 1
    tot_reward = tot_reward + reward

# Visualising the results
Histogram = plt.figure(1)
plt.hist(ID_selected)
plt.title('Thompson Sampling Histogram')
plt.xlabel('IDs')
plt.ylabel('Number of times each ID was selected')
plt.show()
Histogram.savefig(BinPath + '/ThompsonSampling_Histogram_PY.png')