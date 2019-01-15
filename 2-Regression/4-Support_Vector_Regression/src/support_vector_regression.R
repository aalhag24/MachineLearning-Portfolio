# Support Vector Regression

# Path to the file and Libraries
Path = 'MachineLearning-Portfolio/2-Regression/4-Support_Vector_Regression/bin/Position_Salaries.csv'
Path2 = 'MachineLearning-Portfolio/2-Regression/4-Support_Vector_Regression/bin'
#install.packages('caTools', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages("lme4", repos = "http://cran.us.r-project.org",  dependencies = TRUE)
#install.packages('ggplot2', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('e1071', repos = "http://cran.us.r-project.org", dependencies = TRUE)

library(caTools, help, pos = 2, lib.loc = NULL)
library(ggplot2)
library(lme4)
library(e1071)

# Import dataset
dataset = read.csv(Path)
dataset = dataset[2:3]

# Fittin the SVR to the dataset
regressor = svm(formula = Salary ~.,
                data = dataset,
                type = 'eps-regression')

# Predicting a new result
Y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualsing the SVR result
SVR <- ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary),
		color = 'red') +
	geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
		color = 'blue') +
	ggtitle('Truth of Bluff (Linear Regression)') +
	xlab('Level') +
	ylab('Salary')

ggsave("SVRPlot.png",  path = Path2)