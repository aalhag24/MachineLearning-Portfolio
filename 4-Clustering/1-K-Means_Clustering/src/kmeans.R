# K-Means Clustering

# Path to the file and Libraries
###Change File Directory
BinPath = paste(getwd(), 'MachineLearning-Portfolio/4-Clustering/1-K-Means_Clustering/bin', sep='/')
FilePath = paste(BinPath, '/Mall_Customers.csv', sep='')
setwd(BinPath)

install.packages('caTools', repos = "http://cran.us.r-project.org", dependencies = TRUE)
install.packages('cluster',repos = "http://cran.us.r-project.org", dependencies = TRUE)

library(caTools, help, pos = 2, lib.loc = NULL)
library(cluster)

# Import dataset
dataset <- read.csv(FilePath)
X <- dataset[4:5]

# Using the elbow method to find the optimal number of clusters
png("ElbowMethod_R.png") 
set.seed(6)
WCSS <- vector()
for(i in 1:10) WCSS[i] <- sum(kmeans(X, i)$withinss)
ElbowMethodPlot <- plot(1:10, 
                        WCSS, 
                        type = "b", 
                        main = paste('Clusters of clients'), 
                        xlab = "Number of clusters", 
                        ylab = "WCSS")
dev.off() 

# Apply K-means to the dataset
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualising the clusters
png("KMeansClustering_R.png") 
KMeansPlot <- clusplot(X,
                       kmeans$cluster,
                       lines = 0,
                       shade = TRUE,
                       color = TRUE,
                       labels = 2,
                       plotchar = FALSE,
                       span = TRUE,
                       main = paste('Clusters of clients'),
                       xlab = "Annual Income",
                       ylab = "Spending score (1-100)")
dev.off() 