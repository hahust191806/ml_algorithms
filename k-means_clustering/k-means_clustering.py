import numpy as np 
from numpy.random import uniform
import matplotlib.pyplot as plt
import random 
import plotly.express as px
from sklearn.datasets import make_blobs
plt.style.use('dark_background')

    
class KMeans: 
    def __init__(self, X, n_clusters=8):
        self.n_clusters = 8 # cluster number 
        self.max_iters = 300 # max interation. don't want to run inf time
        self.num_examples, self.num_features = X.shape # num of examples, num of features 
        self.plot_figure = True 
        
    # randomly initialize centroids 
    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.n_clusters, self.num_features)) # row, column full with zero 
        for k in range(self.n_clusters):
            centroid = X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid
        return centroids # return random centroids 
    
    # create cluster Function 
    def create_cluster(self, X, centroids):
        clusters = [[] for _ in range(self.n_clusters)] # create n empty list to store points
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point - centroids) ** 2, axis=1)) # centroids: array of centroid
            )
            clusters[closest_centroid].append(point_idx)
        return clusters
    
    # new centroids 
    def calculate_new_centroids(self, clusters, X):
        centroids = np.zeros((self.n_clusters, self.num_features))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid
        return centroids 
    
    # prediction 
    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples) 
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster: 
                y_pred[sample_idx] = cluster_idx
        return y_pred
    
    # fit data 
    def fit(self, X):
        centroids = self.initialize_random_centroids(X) 
        for _ in range(self.max_iters):
            clusters = self.create_cluster(X, centroids)
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)
            diff = centroids - previous_centroids
            if not diff.any():
                break 
        y_pred = self.predict_cluster(clusters, X)
        if self.plot_figure: 
            self.plot_fig(X, y_pred)
        return y_pred 
    
    # ploting scatter plot 
    def plot_fig(self, X, y):
        fig = px.scatter(X[:, 0], X[:, 1], color=y)
        fig.show() # visualize
        
if __name__ == "__main__":
    np.random.seed(10)
    num_clusters = 3 # num of cluster
    X, _ = make_blobs(n_samples=1000, n_features=2, centers=num_clusters) # create dataset using make_blobs from sklearn datasets
    Kmeans = KMeans(X, num_clusters)
    y_pred = Kmeans.fit(X)