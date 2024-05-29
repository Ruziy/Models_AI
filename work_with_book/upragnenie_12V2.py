import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
digits = load_digits()
X = digits.data
y = digits.target

# Create and fit KMeans model
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Assign cluster labels based on the most frequent true label in each cluster
cluster_to_label_map = {}
for i in range(num_clusters):
    labels = y[kmeans.labels_ == i]
    most_common_label = np.argmax(np.bincount(labels))
    cluster_to_label_map[i] = most_common_label

# Predict cluster labels for test data and map them to true labels
test_cluster_indices = kmeans.predict(X)
test_labels = np.array([cluster_to_label_map[i] for i in test_cluster_indices])

# Evaluate accuracy
accuracy = accuracy_score(y, test_labels)
print("Accuracy:", accuracy)