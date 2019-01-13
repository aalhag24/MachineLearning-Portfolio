# Simple Linear Regression

# Path to the file and Libraries
Path = 'MachineLearning-Portfolio/2-Regression/1-Simple_Linear_Regression/bin/Salary_Data.csv'
#install.packages('caTools', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages("tidyverse", repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages("Hmisc", repos = "http://cran.us.r-project.org",  dependencies = TRUE)
#install.packages("lme4", repos = "http://cran.us.r-project.org",  dependencies = TRUE)
#install.packages('ggplot2', repos = "http://cran.us.r-project.org", dependencies = TRUE)
library(caTools, help, pos = 2, lib.loc = NULL)
library(ggplot2)
library(lme4)

# Import dataset
dataset = read.csv(Path)

# Splitting into Training and Test set
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Simple Linear Regression
regressor = lm(formula = Salary ~ YearsExperience, 
				data = training_set)

# Predicting the Test set data
Y_pred = predict(regressor, newdata = test_set)

# Visualization Training data
TrainingPlot <- ggplot() + 
	geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), 
		color = 'red') +
	geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), 
		color = 'blue') + 
	ggtitle('Salary vs Experience(Training set)') +
	xlab('Years of Experience') + 
	ylab('Salary')

print(TrainingPlot)

TestPlot <- ggplot() + 
	geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
		color = 'red') +
	geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), 
		color = 'blue') + 
	ggtitle('Salary vs Experience(Training set)') +
	xlab('Years of Experience') + 
	ylab('Salary')

print(TestPlot)