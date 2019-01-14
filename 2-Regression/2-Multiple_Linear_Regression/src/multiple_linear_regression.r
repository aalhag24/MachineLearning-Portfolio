# Multiple Linear Regression

# Path to the file and Libraries
Path = 'MachineLearning-Portfolio/2-Regression/2-Multiple_Linear_Regression/bin/50_Startups.csv'
#install.packages('caTools', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages("lme4", repos = "http://cran.us.r-project.org",  dependencies = TRUE)
#install.packages('ggplot2', repos = "http://cran.us.r-project.org", dependencies = TRUE)
library(caTools, help, pos = 2, lib.loc = NULL)
library(ggplot2)
library(lme4)

# Import dataset
dataset = read.csv(Path)

# Encode catagorial data
dataset$State = factor(dataset$State,
                        levels = c('New York', 'California', 'Florida'),
                        labels = c(1,2,3))

# Splitting into Training and Test set
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Predict the values
Y_pred = predict(regressor, newdata = test_set)
# print(Y_pred)

# Fitting Multiple Linear Regresion to the Training Set
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data = training_set)
# summary(regressor) # Summarizes the details set above

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
                data = training_set)
# summary(regressor) # check the P-value, remove Administraion

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
                data = training_set)
# summary(regressor) # check the P-value, remove Marketing.Spend
regressor = lm(formula = Profit ~ R.D.Spend,
                data = training_set)
# summary(regressor) 


# Backward Elimination
backwardElimination <- function(x, y, SL) {
    numVar = length(x)
    for(i in c(1:numVar)){
        regressor = lm(formula = Profit ~ ., data = x)
        maxVar = max(coef(summary(regressor))[c(2:numVar), "Pr(>|t|)"])
        if(maxVar > SL){
            j = which(coef(summary(regressor))[c(2:numVar), "Pr(>|t|)"] == maxVar)
            x = x[, -j]
        }
        return(summary(regressor))
    }
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)