# K Means Clustering

# Import the Libraries
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.cluster import KMeans

# Import the data
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
FilePath = BinPath + '\Mall_Customers.csv';
dataset = pd.read_csv(FilePath)

X = dataset.iloc[:, [3,4]].values
###Y = dataset.iloc[:, 4].values

# Using the elbow method to find the opimal number of clusters
WCSS = []
for i in range(1, 11):
	kmeans = KMeans(n_clusters = i, 
					init = 'k-means++', 
					max_iter= 300, 
					n_init= 10, 
					random_state=0)
	kmeans.fit(X)
	WCSS.append(kmeans.inertia_)

# Ploting the Elbow Methods
ElbowMethodPlot = plt.figure(1)
plt.plot(range(1,11), WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
ElbowMethodPlot.show()
ElbowMethodPlot.savefig(BinPath + '\ElbowMethod_PY.png')

# Applying k-means to the mall dataset
kmeans = KMeans(n_clusters=5,
				init='k-means++',
				max_iter=300,
				n_init=10,
				random_state=0)
Y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
KMeansPlot = plt.figure(2)
plt.scatter(X[Y_kmeans == 0, 0], X[Y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[Y_kmeans == 1, 0], X[Y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[Y_kmeans == 2, 0], X[Y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[Y_kmeans == 3, 0], X[Y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[Y_kmeans == 4, 0], X[Y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Cluster of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-10)')
plt.legend()
plt.show()
KMeansPlot.savefig(BinPath + '\KMeansClustering_PY.png')