# Upper Confidence Bound

# Path to the file and Libraries
###Change File Directory
BinPath = paste(getwd(), 'MachineLearning-Portfolio/6-Reinforcement_Learning/1-UCB/bin', sep='/')
FilePath = paste(BinPath, '/Ads_CTR_Optimisation.csv', sep='')
setwd(BinPath)

install.packages('caTools', repos = "http://cran.us.r-project.org", dependencies = TRUE)

library(caTools, help, pos = 2, lib.loc = NULL)

# Import dataset
dataset <- read.csv(FilePath)

# Implamenting UCB
N = 10000
d = 10
num_selection = integer(d)
sum_rewards = integer(d)
ID_selected = integer()
tot_reward = 0

for(n in 1:N){
    ID = 0
    max_upper_bound = 0
    for(i in 1:d){
        if(num_selection[i] > 0) {
            avg_reward = sum_rewards[i] / num_selection[i]
            delta_i = sqrt(3/2 * log(n) / num_selection[i])
            upper_bound = avg_reward + delta_i
        } else {
            upper_bound = 1e100;
        }
        if(upper_bound > max_upper_bound){
            max_upper_bound = upper_bound
            ID = i
        }
    }
    ID_selected = append(ID_selected, ID)
    num_selection[ID] = num_selection[ID] + 1
    reward = dataset[n, ID]
    sum_rewards[ID] = sum_rewards[ID] + reward
    tot_reward = tot_reward + reward
}

# Visualising the results
png("UCB_Histogram_R.png") 
hist(ID_selected,
     col = 'blue',
     main = 'Histogram of ID selection',
     xlab = 'IDs',
     ylab = 'Number of Occurance')
dev.off() 