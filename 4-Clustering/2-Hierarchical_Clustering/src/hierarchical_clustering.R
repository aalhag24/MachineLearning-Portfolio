# Hierarchical Clustering

# Path to the file and Libraries
###Change File Directory
BinPath = paste(getwd(), 'MachineLearning-Portfolio/4-Clustering/2-Hierarchical_Clustering/bin', sep='/')
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
png("Dendrogram_R.png") 
dendrogram = hclust(dist(X, method = 'euclidean'),
                    method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distance')
dev.off() 

# Fitting the cluster to the dataset
hc = hclust(dist(X, method = 'euclidean'),
                    method = 'ward.D')
Y_hc = cutree(hc, 5)

# Visualising the dataset
png("HierarchincalClustering_R.png") 
clusplot(X,
         Y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of clients'),
         xlab = "Annual Income",
         ylab = "Spending Score")
dev.off() 