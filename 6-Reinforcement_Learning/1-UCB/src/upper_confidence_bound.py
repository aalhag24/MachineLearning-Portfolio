# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

# Import the data
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
FilePath = BinPath + '\Ads_CTR_Optimisation.csv';
dataset = pd.read_csv(FilePath)

# Implamenting UCB
N = 10000
d = 10

ID_selected = []
num_selections = [0] * d
sums_rewards = [0] * d
tot_reward = 0

for n in range(0, N):
    max_upper_bound = 0
    ID = 0
    for i in range(0, d):
        if(num_selections[i] > 0):
            avg_reward = sums_rewards[i]/num_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / num_selections[i])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e100;
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ID = i
    ID_selected.append(ID)
    num_selections[ID] = num_selections[ID] + 1
    reward = dataset.values[n, ID]
    sums_rewards[ID] = sums_rewards[ID] + reward
    tot_reward = tot_reward + reward

# Visualising the results
Histogram = plt.figure(1)
plt.hist(ID_selected)
plt.title('Histogram of ID selections')
plt.xlabel('IDs')
plt.ylabel('Number of times each ID was selected')
plt.show()
Histogram.savefig(BinPath + '/UCB_Histogram_PY.png')