# Randon Forest Regression

# Path to the file and Libraries
Path = 'MachineLearning-Portfolio/2-Regression/6-Random_Forest_Regression/bin/Position_Salaries.csv'
BinPath = 'MachineLearning-Portfolio/2-Regression/6-Random_Forest_Regression/bin'
#install.packages('caTools', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('ggplot2', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('randomForest', repos = "http://cran.us.r-project.org", dependencies = TRUE)

library(caTools, help, pos = 2, lib.loc = NULL)
library(ggplot2)
library(randomForest)

# Import dataset
dataset = read.csv(Path)
dataset = dataset[2:3]

# Fittin the SVR to the dataset
set.seed(1234)
regressor = randomForest(x= dataset[1],
                        y = dataset$Salary,
                        ntree = 300)
# dataset[2] returns a dataframe
# dataset$Salary returns a vector

# Predicting a new result
Y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualsing the SVR result
X_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
SVR <- ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary),
		color = 'red') +
	geom_point(aes(x = 6.5, y = Y_pred),
		color = 'black') +
	geom_line(aes(x = X_grid, y = predict(regressor, newdata = data.frame(Level = X_grid))),
		color = 'blue') +
	ggtitle('Truth of Bluff (Randon Forest Regression)') +
	xlab('Level') +
	ylab('Salary')

ggsave("RandomForestRegression_R.png",  path = BinPath)