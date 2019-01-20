# Decision Tree Classification

# Path to the file and Libraries
###Change File Directory
BinPath = 'MachineLearning-Portfolio/3-Classification/6-Decision_Tree_Classification/bin'
FilePath = paste(BinPath, '/Social_Network_Ads.csv', sep='')
#install.packages('caTools', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('ggplot2', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('ElemStatLearn', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('rpart', repos = "http://cran.us.r-project.org", dependencies = TRUE)

library(caTools, help, pos = 2, lib.loc = NULL)
library(ggplot2)
library(ElemStatLearn)
library(rpart)

# Import dataset
dataset = read.csv(FilePath)
dataset = dataset[3:5]

# Encoding the target feature as factor 
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Spliting the dataset into the Training and Testing set
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)        

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting the Classifier on the Training Set
classifier = rpart(formula = Purchased ~ .,
                   data = training_set)

# Predicting a the Testing set
y_pred = predict(classifier, newdata = test_set[-3], type = 'class')

# Create the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualizting the Training set results
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
TrainingPlot <- plot(set[, -3],
                     main = 'DecisionTree Classification (Training set)',
                     xlab = 'Age', ylab = 'Estimated Salary',
                     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
ggsave(file="DecisionTree_Training_R.png", plot=TrainingPlot, path=BinPath)

# Visualising the Test set results
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
TestingPlot <- plot(set[, -3],
                    main = 'DecisionTree Classification (Testing set)',
                    xlab = 'Age', ylab = 'Estimated Salary',
                    xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
ggsave(file="DecisionTree_Testing_R.png", plot=TestingPlot, path = BinPath)


# Ploting the Decision Tree
plot(classifier)
text(classifier)