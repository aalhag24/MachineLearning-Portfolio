# Regression Template

# Path to the file and Libraries
Path = 'MachineLearning-Portfolio/2-Regression/3-Polynomial_Regression/bin/Position_Salaries.csv'
Path2 = 'MachineLearning-Portfolio/2-Regression/3-Polynomial_Regression/bin'
library(caTools, help, pos = 2, lib.loc = NULL)
library(ggplot2)
library(lme4)

# Import dataset
dataset = read.csv(Path)
dataset = dataset[2:3]

# Splitting into Training and Test set
#set.seed(123)
#split = sample.split(dataset$Profit, SplitRatio = 0.8)
#training_set = subset(dataset, split == TRUE)
#test_set = subset(dataset, split == FALSE)

# Feature Scaling

# Fitting Poynomial Regression to the dataset
###Create your regression model
poly_reg = lm(formula = Salary ~.,
            data = dataset)

# Predict a new result with Regression
###Edit these values
Y_pred = predict(regressor, data.frame(Level = 6.5))
print(Y_pred)

# Visualising the Regression result
X_grid = seq(min(dataset$Level), max(datase$Level), 0.1)
LinearPlot <- ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary),
		color = 'red') +
	geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = data.frame(Level = X_grid))),
		color = 'blue') +
	ggtitle('Truth of Bluff (Linear Regression)') +
	xlab('Level') +
	ylab('Salary')

ggsave("LinearPlot.png",  path = Path2)