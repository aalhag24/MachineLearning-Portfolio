# Natural Language Processing

# Path to the file and Libraries
BinPath = paste(getwd(), 'MachineLearning-Portfolio/6-Reinforcement_Learning/1-UCB/bin', sep='/')
FilePath = paste(BinPath, '/Restaurant_Reviews.tsv', sep='')
setwd(BinPath)

install.packages('tm', repos = "http://cran.us.r-project.org", dependencies = TRUE)
install.packages('SnowballC', repos = "http://cran.us.r-project.org", dependencies = TRUE)

# Import dataset
dataset_original <- read.delim(FilePath, quote = '', stringsAsFactors = FALSE)

# Cleaning the text
library(SnowballC)
library(tm)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Encoding the target feature as factor 
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Spliting the dataset into the Training and Testing set
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)        

# Fitting the Classifier on the Training Set
classifier = randomForest(x = training_set[-692],
                   y = training_set$Liked,
                   ntree = 10)

# Predicting a the Testing set
y_pred = predict(classifier, newdata = test_set[-692], type = 'class')

# Create the Confusion Matrix
cm = table(test_set[, 692], y_pred)

# Evaluate the performance of each of these models. 
# (TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives)

# Accuracy = (TP + TN) / (TP + TN + FP + FN)

#       Precision (measuring exactness) 
# Precision = TP / (TP + FP)

#       Recall (measuring completeness) 
# Recall = TP / (TP + FN)

#       F1 Score (compromise between Precision and Recall)
# F1_Score = 2 * Precision * Recall / (Precision + Recall) */