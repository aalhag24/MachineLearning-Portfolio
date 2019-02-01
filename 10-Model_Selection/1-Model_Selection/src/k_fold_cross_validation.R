# K-Fold Cross Validation

# Path to the file and Libraries
BinPath = 'MachineLearning-Portfolio/10-Model_Selection/1-Model_Selection/bin'
FilePath = paste(BinPath, '/Social_Network_Ads.csv', sep='')

#install.packages('caTools', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('ggplot2', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('ElemStatLearn', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('e1071', repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('caret', repos = "http://cran.us.r-project.org", dependencies = TRUE)

library(caTools, help, pos = 2, lib.loc = NULL)
library(ggplot2)
library(ElemStatLearn)
library(e1071)

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
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernal = 'radial')

# Predicting a the Testing set
y_pred = predict(classifier, type = 'response', newdata = test_set[-3])

# Create the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Creating the k-Fold Cross Validation
library(caret)
folds = createFolds(training_set$Purchased, k = 10)
cv = lapply(folds, function(x){
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = Purchased ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernal = 'radial')
  y_pred = predict(classifier, type = 'response', newdata = test_fold[-3])
  cm = table(test_fold[, 3], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})

accuracy = mean(as.numeric(cv))


# Visualizting the Training set results
setwd(BinPath)
png("K_Fold_Cross_Validation_Training_R.png") 
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, type = 'response', newdata = grid_set)
TrainingPlot <- plot(set[, -3],
                     main = 'Non-Linear Kernal SVM (Training set)',
                     xlab = 'Age', ylab = 'Estimated Salary',
                     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
dev.off() 

# Visualising the Test set results
png("K_Fold_Cross_Validation_Testing_R.png") 
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, type = 'response', newdata = grid_set)
TestingPlot <- plot(set[, -3],
                    main = 'Non-Linear Kernal SVM (Testing set)',
                    xlab = 'Age', ylab = 'Estimated Salary',
                    xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
dev.off() 