# Hierarchical Clustering

# Import the Libraries
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Import the data
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
FilePath = BinPath + '\Mall_Customers.csv';
dataset = pd.read_csv(FilePath)

X = dataset.iloc[:, [3,4]].values

# Using the dendrogram to find the opimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
Dendro = plt.figure(1)
plt.title("Dendrogram")
plt.xlabel('Customers')
plt.ylabel('Euclidian Distance')
Dendro.show()
Dendro.savefig(BinPath + '\Dendrogram_PY.png')

# Fitting heirarchical clustering to the mall dataset
hc = AgglomerativeClustering(n_clusters=5,
							 affinity= 'euclidean',
							 linkage='ward')
Y_hc = hc.fit_predict(X)

# Visualising the clusters
HierarchicalClusteringPlot = plt.figure(2)
plt.scatter(X[Y_hc == 0, 0], X[Y_hc == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[Y_hc == 1, 0], X[Y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[Y_hc == 2, 0], X[Y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[Y_hc == 3, 0], X[Y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[Y_hc == 4, 0], X[Y_hc == 4, 1], s = 100, c = 'magenta', label= 'Sensible')
plt.title('Cluster of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-10)')
plt.legend()
plt.show()
HierarchicalClusteringPlot.savefig(BinPath + '\HierarchincalClustering_PY.png')