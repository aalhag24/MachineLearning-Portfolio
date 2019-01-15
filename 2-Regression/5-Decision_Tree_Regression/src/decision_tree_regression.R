# Decision Tree Regression

# Path to the file and Libraries
Path = 'MachineLearning-Portfolio/2-Regression/5-Decision_Tree_Regression/bin/Position_Salaries.csv'
BinPath = 'MachineLearning-Portfolio/2-Regression/5-Decision_Tree_Regression/bin'
#install.packages('caTools', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages("lme4", repos = "http://cran.us.r-project.org",  dependencies = TRUE)
#install.packages('ggplot2', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('e1071', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('rpart', repos = "http://cran.us.r-project.org", dependencies = TRUE)

library(caTools, help, pos = 2, lib.loc = NULL)
library(ggplot2)
library(lme4)
library(rpart)

# Import dataset
dataset = read.csv(Path)
dataset = dataset[2:3]

# Fittin the SVR to the dataset
regressor = rpart(formula = Salary ~.,
                data = dataset,
				control = rpart.control(minsplit = 1))

# Predicting a new result
Y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualsing the SVR result
X_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
SVR <- ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary),
		color = 'red') +
	geom_point(aes(x = 6.5, y = Y_pred),
		color = 'green') +
	geom_line(aes(x = X_grid, y = predict(regressor, newdata = data.frame(Level = X_grid))),
		color = 'blue') +
	ggtitle('Truth of Bluff (Decision Tree Regression)') +
	xlab('Level') +
	ylab('Salary')

ggsave("DecisionTreeRegression_R.png",  path = BinPath)