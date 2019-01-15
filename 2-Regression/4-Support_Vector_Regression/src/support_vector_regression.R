# Support Vector Regression

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
