# Artifical Neural Network

# Path to the file and Libraries
BinPath = paste('MachineLearning-Portfolio/8-Deep_Learning/1-Artificial_Neural_Networks/bin', sep='/')
FilePath = paste(BinPath, '/Churn_Modelling.csv', sep='')

install.packages('caTools', repos = "http://cran.us.r-project.org", dependencies = TRUE)
install.packages('h2o', repos = "http://cran.us.r-project.org", dependencies = TRUE)

# Import dataset
dataset = read.csv(FilePath)
dataset = dataset[4:14]

# Encoding the categorical variables
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting the ANN classifier to the training set
library(h2o)
h2o.init(nthreads = -1)
model = h2o.deeplearning(y = 'Exited',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(5,5),
                         epochs = 100,
                         train_samples_per_iteration = -2)

# Predicting the Test set
Y_pred = h2o.predict(model, newdata = as.h2o(test_set[-11]))
Y_pred = (Y_pred > 0.5)
Y_pred = as.vector(Y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], Y_pred)

# Shutdown the server
h2o.shutdown()