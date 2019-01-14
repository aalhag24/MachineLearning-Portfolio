# Polynomial Linear Regression

# Path to the file and Libraries
Path = 'MachineLearning-Portfolio/2-Regression/3-Polynomial_Regression/bin/Position_Salaries.csv'
Path2 = 'MachineLearning-Portfolio/2-Regression/3-Polynomial_Regression/bin'
#install.packages('caTools', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages("lme4", repos = "http://cran.us.r-project.org",  dependencies = TRUE)
#install.packages('ggplot2', repos = "http://cran.us.r-project.org", dependencies = TRUE)
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

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~.,
            data = dataset)

# Fitting Poynomial Regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4

poly_reg = lm(formula = Salary ~.,
            data = dataset)

# Visualising the Linear Regression
LinearPlot <- ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary),
		color = 'red') +
	geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
		color = 'blue') +
	ggtitle('Truth of Bluff (Linear Regression)') +
	xlab('Level') +
	ylab('Salary')

ggsave("LinearPlot.png",  path = Path2)


# Visualising the Polynomial Regression
PolynomialPlot <- ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary),
		color = 'red') +
	geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
		color = 'blue') +
	ggtitle('Truth of Bluff (Polynomial Regression)') +
	xlab('Level') +
	ylab('Salary')

ggsave("PolynomialPlot.png",  path = Path2)

# Predict a new result with Linear Regression
Y_pred = predict(lin_reg, data.frame(Level = 6.5))
print(Y_pred)

# Predict a new result with Polynomial Regression
Y_pred = predict(poly_reg, data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3, Level4 = 6.5^4))
print(Y_pred)