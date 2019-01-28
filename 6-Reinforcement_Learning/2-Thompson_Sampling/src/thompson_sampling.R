# Thompson Sampling

# Path to the file and Libraries
###Change File Directory
BinPath = paste(getwd(),'MachineLearning-Portfolio/6-Reinforcement_Learning/2-Thompson_Sampling/bin', sep='/')
FilePath = paste(BinPath, '/Ads_CTR_Optimisation.csv', sep='')
setwd(BinPath)

#install.packages('caTools', repos = "http://cran.us.r-project.org", dependencies = TRUE)

library(caTools, help, pos = 2, lib.loc = NULL)

# Import dataset
dataset = read.csv(FilePath)

# Implamenting Thompson Sampling
N = 10000
d = 10
tot_reward = 0

ID_selected = integer(0)
num_rewards_1 = integer(d)
num_rewards_0 = integer(d)

for(n in 1:N){
    ID = 0
    max_rand = 0
    for(i in 1:d){
        beta_rand = rbeta(n = 1,
                          shape1 = num_rewards_1[i] + 1,
                          shape2 = num_rewards_0[i] + 1)
        if(beta_rand > max_rand){
            max_rand = beta_rand
            ID = i
        }
    }
    ID_selected = append(ID_selected, ID)
    reward = dataset[n, ID]
    if(reward == 1){
        num_rewards_1[ID] = num_rewards_1[ID] + 1
    } else {
        num_rewards_0[ID] = num_rewards_0[ID] + 1
    }
    tot_reward = tot_reward + reward
}

print(tot_reward)

# Visualising the results
png("ThompsonSampling_Histogram_R.png") 
hist(ID_selected,
     col = 'blue',
     main = 'Thompson Sampling Histogram of ID selection',
     xlab = 'IDs',
     ylab = 'Number of Occurance')
dev.off() 