# Importing necessary libraries
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris 
import matplotlib.pyplot as plt


# Using only two features (sepal length and sepal width) for simplicity
iris = load_iris()
X = iris.data[:, :2]  # Selecting first two features (sepal length and sepal width)

# Initialize the K-means model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model to the data
kmeans.fit(X)

# Predict the cluster labels
y_pred = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('K-means Clustering (Predicted Clusters)')
plt.show()
