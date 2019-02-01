# XGBoost

# Path to the file and Libraries
BinPath = 'MachineLearning-Portfolio/10-Model_Selection/2-XGBoost/bin'
FilePath = paste(BinPath, '/Churn_Modelling.csv', sep='')

#install.packages('caTools', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('ggplot2', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('ElemStatLearn', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('e1071', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('xgboost', repos = "http://cran.us.r-project.org", dependencies = TRUE)

library(caTools, help, pos = 2, lib.loc = NULL)
library(ggplot2)
library(ElemStatLearn)
library(e1071)

# Import dataset
dataset = read.csv(FilePath)
dataset = dataset[4:14]

# Encoding the categorical variables
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting XGBoost to the Training set
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-11]), 
                     label = training_set$Exited,
                     nrounds = 10)

# Creating the k-Fold Cross Validation
library(caret)
folds = createFolds(training_set$Exited, k = 10)
cv = lapply(folds, function(x){
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = xgboost(data = as.matrix(training_set[-11]), 
                       label = training_set$Exited,
                       nrounds = 10)
  y_pred = predict(classifier, type = 'response', newdata = as.matrix(test_fold[-11]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 11], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))