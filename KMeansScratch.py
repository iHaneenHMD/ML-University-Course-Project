# clustering_base.py

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

class clustering_base:
    def __init__(self):
        pass

    def initialize_algorithm(self):
        pass

    def fit(self,X):
        pass

    def get_index(self,nlevel,cluster_number):
        # nlevel (integer)
        # cluster_number (integer)
        # return indices of samples where clustersave[nlevel] = cluster_number
        return np.where(np.absolute(self.clustersave[nlevel]-cluster_number)<1e-5)[0]

    def plot_objective(self,title="",xlabel="",ylabel=""):
        # plot objective function if data is collected
        if len(self.objectivesave)>0:
            fig = plt.figure()
            plt.plot(self.objectivesave,'b-',marker="o",markersize=5)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

    def plot_cluster(self,nlevel=-1,title="",xlabel="",ylabel=""):
        # plot cluster assignment for dataset for self.clustersave[nlevel]
        fig,ax = plt.subplots(1,1)
        # plot data points separate color for each cluster
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        color = (self.clustersave[nlevel]+1)/self.ncluster
        scat = ax.scatter(self.X[0,:],self.X[1,:],color=cm.jet(color),marker="o",s=15)

    def plot_cluster_animation(self,nlevel=-1,interval=50,title="",xlabel="",ylabel=""):
        # create animation for cluster assignments in self.clustersave[level]
        # for level = 0,1,...,nlevel
        # interval is the time (in milliseconds) between frames in animation
        fig,ax = plt.subplots(1,1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        nframe = len(self.clustersave)
        if nlevel < 0:
            nframe = nframe + 1 + nlevel
        else:
            nframe = nlevel
        # scatter plot all data points in same color
        scat = ax.scatter(self.X[0,:],self.X[1,:],color=cm.jet(0),marker="o",s=15)
        # update function for animation change color according to cluster assignment
        def update(i,scat,clustersave,ncluster):
            array_color_data = (1+self.clustersave[i])/(self.ncluster+1e-16)
            scat.set_color(cm.jet(array_color_data))
            return scat,
        # create animation
        ani = animation.FuncAnimation(fig=fig, func=update, frames = nframe,
            fargs=[scat,self.clustersave,self.ncluster], repeat_delay=5000, repeat=True, interval=interval, blit=True)
        # uncomment to create mp4
        # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get detaisl
        #ani.save('Clustering_Animation.mp4', writer='ffmpeg')
        return ani


import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
from clustering_Base import  clustering_base

class KMeansScratch(clustering_base):

    def __init__(self, n_clusters, max_iterations=300):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = None

        #new add after fiting
        self.clustersave = []
        self.objectivesave = []

    def fit(self, data):

        #new add after fiting
        self.X = data  # Storing data for plotting purposes

        #old
        self.centroids = self._initialize_centroids(data)
        for _ in range(self.max_iterations):
            assigned_clusters = self._assign_clusters(data, self.centroids)
            new_centroids = self._update_centroids(data, assigned_clusters)

            if self._check_convergence(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

            #new add after fiting
            self.clustersave.append(assigned_clusters)  # Save cluster assignments for plotting


        #same old
    def _initialize_centroids(self, data):
        centroids_indices = np.random.choice(len(data), self.n_clusters, replace=False)
        centroids = data[centroids_indices]
        return centroids

       #same old
    def _assign_clusters(self, data, centroids):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
       #same old
    def _update_centroids(self, data, assigned_clusters):
        new_centroids = np.zeros((self.n_clusters, data.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = data[assigned_clusters == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = centroids[i]
        return new_centroids

       #same old
    def _check_convergence(self, centroids, new_centroids, tol=1e-4):
        return np.linalg.norm(new_centroids - centroids) < tol

#End of class KMeansScratch(clustering_base):

''''
Create an instance of the KMeansScratch class,
specifying the number of clusters , and then call the fit method, passing your dataset as an argument.
For instance:
'''
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets as skdatasets
from sklearn.preprocessing import StandardScaler

def generate_dataset(n_samples, case, random_state):
    if case == "blobs":
        X, _ = skdatasets.make_blobs(n_samples=n_samples, random_state=random_state)
    elif case == "aniso":
        X, _ = skdatasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
    elif case == "noisy_moons":
        X, _ = skdatasets.make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
    elif case == "noisy_circles":
        X, _ = skdatasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
    else:
        raise ValueError("Invalid case parameter.")

    return X

# Set parameters
n_samples = 300
#2+1+1+1+2+5+1.=13 2+0+0+6+9+0+4. =21 2+1+1+1+7+1+6. =19 2+1+1+7+0+2+2.=15  2+1+1+6+9+2+2=23 Total 91
# Sum of the individual digits from the student IDs
random_state =91

# Define the desired datasets
dataset_names = ["blobs", "aniso", "noisy_moons", "noisy_circles"]

# Generate and plot the datasets
fig, ax = plt.subplots(2, 2)
for i, dataset_name in enumerate(dataset_names):
    X = generate_dataset(n_samples, dataset_name, random_state)
    row = i // 2
    col = i % 2
    ax[row, col].scatter(X[:, 0], X[:, 1], color=cm.jet(0), s=15)
    ax[row, col].set_title(f"Dataset{i+1}: {dataset_name.capitalize()}")

plt.tight_layout()
plt.show()

# Import the clustering algorithm
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
from clustering_Base import  clustering_base
from  KMeansScratch import KMeansScratch

# Define the number of clusters
n_clusters = 2

# Create instances of the KMeansScratch class for each dataset
kmeans_dataset1 = KMeansScratch(n_clusters=n_clusters)
kmeans_dataset2 = KMeansScratch(n_clusters=n_clusters)
kmeans_dataset3 = KMeansScratch(n_clusters=n_clusters)
kmeans_dataset4 = KMeansScratch(n_clusters=n_clusters)

# Apply the clustering algorithm to each dataset
dataset1 = generate_dataset(n_samples, dataset_names[0], random_state)
dataset2 = generate_dataset(n_samples, dataset_names[1], random_state)
dataset3 = generate_dataset(n_samples, dataset_names[2], random_state)
dataset4 = generate_dataset(n_samples, dataset_names[3], random_state)

kmeans_dataset1.fit(dataset1)
kmeans_dataset2.fit(dataset2)
kmeans_dataset3.fit(dataset3)
kmeans_dataset4.fit(dataset4)


# Visualize the clustering results for each dataset
kmeans_dataset1.plot_cluster(title="Clustering Results for Dataset 1", xlabel="Feature 1", ylabel="Feature 2")
plt.show()

kmeans_dataset2.plot_cluster(title="Clustering Results for Dataset 2", xlabel="Feature 1", ylabel="Feature 2")
plt.show()

kmeans_dataset3.plot_cluster(title="Clustering Results for Dataset 3", xlabel="Feature 1", ylabel="Feature 2")
plt.show()

kmeans_dataset4.plot_cluster(title="Clustering Results for Dataset 4", xlabel="Feature 1", ylabel="Feature 2")
plt.show()

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets as skdatasets  # Add this line
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from clustering_Base import clustering_base
from KMeansScratch import KMeansScratch

def generate_dataset_with_labels(n_samples, case, random_state):
    # Your dataset generation code here
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
def evaluate_clustering(true_labels, predicted_labels):
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    rand_stat = adjusted_rand_score(true_labels, predicted_labels)
    return f1, nmi, rand_stat


# Generate and plot the datasets
fig, ax = plt.subplots(2, 2)

for i, dataset_name in enumerate(dataset_names):
    X, true_labels = generate_dataset_with_labels(n_samples, dataset_name, random_state)

    # Apply your KMeansScratch clustering
    kmeans = KMeansScratch(n_clusters=3)
    kmeans.fit(X)
    predicted_labels = kmeans.clustersave[-1]  # Get the last cluster assignment

    # Evaluate clustering
    f1, nmi, rand_stat = evaluate_clustering(true_labels, predicted_labels)

    # Plotting
    row = i // 2
    col = i % 2
    ax[row, col].scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', s=15)
    ax[row, col].set_title(f"Dataset{i+1}: {dataset_name.capitalize()}\nF1: {f1:.2f}, NMI: {nmi:.2f}, Rand: {rand_stat:.2f}")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets as skdatasets  # Add this line
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from clustering_Base import clustering_base
from KMeansScratch import KMeansScratch

def generate_dataset_with_labels(n_samples, case, random_state):
    # Your dataset generation code here
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
def evaluate_clustering(true_labels, predicted_labels):
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    rand_stat = adjusted_rand_score(true_labels, predicted_labels)
    return f1, nmi, rand_stat

