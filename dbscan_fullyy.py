import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib import cm
from sklearn import datasets as skdatasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        self.labels = np.full(X.shape[0], -1)
        cluster_id = 0

        for i in range(X.shape[0]):
            if self.labels[i] != -1:
                continue

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                self.labels[i] = 0  # Noise point
            else:
                cluster_id += 1
                self._expand_cluster(X, i, neighbors, cluster_id)

        return self.labels

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels[point_idx] = cluster_id

        while neighbors:
            current_point = neighbors.pop(0)

            if self.labels[current_point] == -1:
                new_neighbors = self._get_neighbors(X, current_point)

                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)

            if self.labels[current_point] == -1 or self.labels[current_point] == 0:  # Not assigned or noise point
                self.labels[current_point] = cluster_id

    def _get_neighbors(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return [i for i in range(len(distances)) if distances[i] <= self.eps]

    def find_best_eps(self, data):
        min_points = self.min_samples #minpts
        kn = NearestNeighbors(n_neighbors=min_points).fit(data) #fit the model knn with k=min points
        distance, idx = kn.kneighbors(data) # distances from each data point to its k nearest neighbors.  and the indices of the k nearest neighbors for each data point.
        distance2 = sorted(distance[:, min_points-1], reverse=True) #sort the distances
        plt.plot(list(range(1, len(distance2)+1)), distance2) #plot it
        plt.show()

def generate_dataset(n_samples, case, random_state):
    if case == "blobs":
        X, labels = skdatasets.make_blobs(n_samples=n_samples, random_state=random_state)
    elif case == "aniso":
        X, labels = skdatasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
    elif case == "noisy_moons":
        X, labels = skdatasets.make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
    elif case == "noisy_circles":
        X, labels = skdatasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
    else:
        raise ValueError("Invalid case parameter.")

    return X, labels

# Set parameters
n_samples = 300
random_state = 91

# Define the desired datasets
dataset_names = ["blobs", "aniso", "noisy_moons", "noisy_circles"]

# Generate and plot the datasets
fig, ax = plt.subplots(2, 2)
for i, dataset_name in enumerate(dataset_names):
    X, true_labels = generate_dataset(n_samples, dataset_name, random_state)
    row = i // 2
    col = i % 2
    ax[row, col].scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', s=15)
    ax[row, col].set_title(f"Dataset{i+1}: {dataset_name.capitalize()}")

plt.tight_layout()
plt.show()

"""# ****case1****"""

dataset_name = "blobs"
data, true_labels = generate_dataset(300, 'blobs', 91)
plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap="viridis")
plt.title(f"Dataset: {dataset_name.capitalize()}")
plt.show()

db=DBSCAN(None,min_samples=5)
db.find_best_eps(data)

db.eps=0.8
predicted_labels=db.fit(data)

np.unique(predicted_labels)

plt.scatter(data[:,0] , data[:,1], c =predicted_labels,cmap = "viridis");

plt.figure(figsize=(12, 5))

# Plot the true labels
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis')
plt.title('True Labels')

# Plot the predicted labels
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
plt.title('Predicted Labels')

plt.show()

"""# ****showing the accuracy of the model using dataset1****"""

from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import silhouette_score

# F-measure
f_measure = f1_score(true_labels, predicted_labels, average='macro')

# Normalized Mutual Information
nmi = normalized_mutual_info_score(true_labels, predicted_labels)

# Rand Statistic
rand_statistic = adjusted_rand_score(true_labels, predicted_labels)

#silhouette average
silhouette_avg = silhouette_score(data, predicted_labels)

# Print the results
print(f"F-measure: {f_measure:.4f}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")
print(f"Rand Statistic: {rand_statistic:.4f}")
print("Silhouette Score:", silhouette_avg)
"""In summary, while the F-measure suggests some room for improvement, the Silhouette Score, Normalized Mutual Information, and Rand Statistic indicate good overall performance of the clustering algorithm.

# ****case2****
"""

dataset_name = "aniso"
data, true_labels = generate_dataset(300, 'aniso', 91)
plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap="viridis")
plt.title(f"Dataset: {dataset_name.capitalize()}")
plt.show()

db.find_best_eps(data)

db.eps=0.5
predicted_labels=db.fit(data)

np.unique(predicted_labels)

plt.scatter(data[:,0] , data[:,1], c =predicted_labels,cmap = "viridis");

plt.figure(figsize=(12, 5))

# Plot the true labels
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis')
plt.title('True Labels')

# Plot the predicted labels
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
plt.title('Predicted Labels')

plt.show()

"""# ****showing the accuracy of the model using dataset2****"""

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import precision_score, recall_score, f1_score

# F-measure
f_measure = f1_score(true_labels, predicted_labels, average='macro')

# Normalized Mutual Information
nmi = normalized_mutual_info_score(true_labels, predicted_labels)

# Rand Statistic
rand_statistic = adjusted_rand_score(true_labels, predicted_labels)

#silhouette average
silhouette_avg = silhouette_score(data, predicted_labels)

# Print the results
print(f"F-measure: {f_measure:.4f}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")
print(f"Rand Statistic: {rand_statistic:.4f}")

"""# ****case3****"""

dataset_name = "noisy_moons"
data, true_labels = generate_dataset(300, 'noisy_moons', 91)
plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap="viridis")
plt.title(f"Dataset: {dataset_name.capitalize()}")
plt.show()

db.find_best_eps(data)

db.eps=0.155
predicted_labels=db.fit(data)

np.unique(predicted_labels)

plt.scatter(data[:,0] , data[:,1], c =predicted_labels,cmap = "viridis");

plt.figure(figsize=(12, 5))

# Plot the true labels
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis')
plt.title('True Labels')

# Plot the predicted labels
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
plt.title('Predicted Labels')

plt.show()

"""# ****showing the accuracy of the model using dataset3****"""

# F-measure
f_measure = f1_score(true_labels, predicted_labels, average='macro')

# Normalized Mutual Information
nmi = normalized_mutual_info_score(true_labels, predicted_labels)

# Rand Statistic
rand_statistic = adjusted_rand_score(true_labels, predicted_labels)

#silhouette average
silhouette_avg = silhouette_score(data, predicted_labels)

# Print the results
print(f"F-measure: {f_measure:.4f}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")
print(f"Rand Statistic: {rand_statistic:.4f}")

"""# ****case4****"""

dataset_name = "noisy_circles"
data, true_labels = generate_dataset(300, 'noisy_circles', 91)
plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap="viridis")
plt.title(f"Dataset: {dataset_name.capitalize()}")
plt.show()

db.find_best_eps(data)

db.eps=0.157
db.min_samples=4
predicted_labels=db.fit(data)

np.unique(predicted_labels)

plt.scatter(data[:,0] , data[:,1], c =predicted_labels,cmap = "viridis");

plt.figure(figsize=(12, 5))

# Plot the true labels
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis')
plt.title('True Labels')

# Plot the predicted labels
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
plt.title('Predicted Labels')

plt.show()

"""# ****showing the accuracy of the model using dataset4****"""

# F-measure
f_measure = f1_score(true_labels, predicted_labels, average='macro')

# Normalized Mutual Information
nmi = normalized_mutual_info_score(true_labels, predicted_labels)

# Rand Statistic
rand_statistic = adjusted_rand_score(true_labels, predicted_labels)

#silhouette average
silhouette_avg = silhouette_score(data, predicted_labels)

# Print the results
print(f"F-measure: {f_measure:.4f}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")
print(f"Rand Statistic: {rand_statistic:.4f}")


