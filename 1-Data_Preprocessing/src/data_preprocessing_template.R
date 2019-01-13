# Data preprocessing

# Path to the file and Libraries
Path = 'MachineLearning-Portfolio/1-Data_Preprocessing/bin/Data.csv'
#install.packages('caTools', repos = "http://cran.us.r-project.org")
library(caTools, help, pos = 2, lib.loc = NULL)

# Import dataset
dataset = read.csv(Path)

# Remore missing data
dataset$Age = ifelse(is.na(dataset$Age), 
                    ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                    dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary), 
                    ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                    dataset$Salary)

# Encode catagorial data
dataset$Country = factor(dataset$Country,
                        levels = c('France', 'Spain', 'Germany'),
                        labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                        levels = c('No', 'Yes'),
                        labels = c(0,1))

# Splitting into Training and Test set
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])

# Show data
print(dataset)
print(split)