# Principle Component Analysis

# Path to the file and Libraries
BinPath = '9-Dimenstionality_Reduction/Kernel_PCA/bin'
FilePath = paste(BinPath, '/Social_Network_Ads.csv', sep='');

install.packages('caret', repos = "http://cran.us.r-project.org", dependencies = TRUE)
install.packages('e1071', repos = "http://cran.us.r-project.org", dependencies = TRUE)
install.packages('ggplot2', repos = "http://cran.us.r-project.org", dependencies = TRUE)
install.packages('ElemStatLearn', repos = "http://cran.us.r-project.org", dependencies = TRUE)
install.packages('kernlab', repos = "http://cran.us.r-project.org", dependencies = TRUE)

library(caret, help, pos = 2, lib.loc = NULL)
library(ggplot2)
library(e1071)
library(ElemStatLearn)

# Import dataset
dataset = read.csv(FilePath)
dataset = dataset[, 3:5]

# Spliting the dataset into the Training and Testing set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[, 1:2] = scale(training_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])

# Creating the kernel pca
library(kernlab)
kpca = kpca(~., data = training_set[-3], kernel = 'rbfdot', features = 2)
training_set_pca = as.data.frame(predict(kpca, training_set))
training_set_pca$Purchased = training_set$Purchased
test_set_pca = as.data.frame(predict(kpca, test_set))
test_set_pca$Purchased = test_set$Purchased

# Fitting the SVM Classifier on the Training Set
classifier = svm(formula = Purchased ~ .,
                 data = training_set_pca,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting a the Testing set
y_pred = predict(classifier, newdata = test_set_pca[-3])

# Create the Confusion Matrix
cm = table(test_set_pca[, 3], y_pred)

# Visualizting the Training set results
setwd(BinPath)
png("KernelPCA_Training_R.png")
set = training_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel PCA (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))
dev.off()

# Visualising the Test set results
png("KernelPCA_Testing_R.png")
set = test_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel PCA (Testing set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))
dev.off()